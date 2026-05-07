"""
Catalyst Generator — Three Conditional Generation Models
=========================================================

Uses the pretrained GIN encoder + SELFIES decoder from the molecular
representation learning pipeline. Three separate catalyst predictors are
trained on different input conditioning signals:

  Model A: reactants + solvents  → catalyst SELFIES
  Model B: reactants + products  → catalyst SELFIES
  Model C: reactants + products + solvents → catalyst SELFIES

Architecture per model
----------------------
  1. GIN encoder (frozen / later finetuned) encodes each input molecule.
  2. Input embeddings are mean-pooled into a single context vector.
  3. CatalystTransformer (the *trainable* middle layer) maps context → a
     sequence of catalyst latent vectors.
  4. Each latent vector is decoded by the SELFIES decoder into one catalyst
     molecule.

Training schedule
-----------------
  Phase 1  (epochs 1-5):  GIN + Decoder FROZEN — only CatalystTransformer trains.
  Phase 2  (epochs 6-10): Full end-to-end finetuning at a reduced LR.

Checkpoints are saved after every epoch.
"""

import os
import ast
import math
import random
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

import selfies as sf
from transformers import AutoTokenizer, AutoModel

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool

# ============================================================
# RE-IMPORT SHARED COMPONENTS FROM MAIN PIPELINE
# (copy-paste the classes / helpers you need, or import them)
# ============================================================

# ---- CONFIG (keep in sync with main pipeline) ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
LATENT_DIM   = 768
GIN_HIDDEN   = 300
GIN_LAYERS   = 5
ATOM_FEATS   = 9

BATCH_SIZE   = 16          # smaller batch — multiple molecules per row
LR_PHASE1    = 3e-4        # CatalystTransformer only
LR_PHASE2    = 5e-5        # full finetune
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 5
MAX_LEN      = 128
MAX_CATALYSTS = 4          # max number of catalyst SELFIES to predict

# Transformer middle layer
CATGEN_D_MODEL   = 512     # hidden dim of the catalyst transformer (≥ 512)
CATGEN_NHEAD     = 8
CATGEN_LAYERS    = 6
CATGEN_FF_DIM    = 2048
CATGEN_DROPOUT   = 0.1

SAVE_DIR_BASE = "checkpoints_catalyst"
os.makedirs(SAVE_DIR_BASE, exist_ok=True)

# ---- ChemBERTa (frozen) ----
print(f"Loading ChemBERTa: {CHEMBERTA_MODEL}")
chemberta_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
chemberta_model     = AutoModel.from_pretrained(CHEMBERTA_MODEL).to(DEVICE)
for p in chemberta_model.parameters():
    p.requires_grad = False
chemberta_model.eval()
print("ChemBERTa loaded ✓")

@torch.no_grad()
def embed_smiles_batch(smiles_list: list[str]) -> torch.Tensor:
    """Encode a list of SMILES → (B, LATENT_DIM) using frozen ChemBERTa."""
    enc = chemberta_tokenizer(
        smiles_list, return_tensors="pt",
        padding=True, truncation=True, max_length=256,
    ).to(DEVICE)
    out = chemberta_model(**enc)
    return out.last_hidden_state[:, 0, :]          # CLS token

# ============================================================
# ATOM FEATURES + SMILES → GRAPH
# ============================================================

HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP:    0,
    Chem.rdchem.HybridizationType.SP2:   1,
    Chem.rdchem.HybridizationType.SP3:   2,
    Chem.rdchem.HybridizationType.SP3D:  3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
    Chem.rdchem.HybridizationType.OTHER: 5,
}

def atom_features(atom) -> list:
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        HYBRIDIZATION_MAP.get(atom.GetHybridization(), 5),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetMass() / 100.0,
        atom.GetTotalValence(),
    ]

def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    if not edges:
        edges = [[0, 0]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# ============================================================
# GIN ENCODER  (same as main pipeline)
# ============================================================

class GINEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(GIN_LAYERS):
            in_dim = ATOM_FEATS if i == 0 else GIN_HIDDEN
            mlp    = nn.Sequential(
                nn.Linear(in_dim, GIN_HIDDEN * 2), nn.ReLU(),
                nn.Linear(GIN_HIDDEN * 2, GIN_HIDDEN),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(GIN_HIDDEN))
        self.project = nn.Sequential(
            nn.Linear(GIN_HIDDEN, LATENT_DIM * 2), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(LATENT_DIM * 2, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
        )

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        x = global_mean_pool(x, batch_idx)
        return self.project(x)

# ============================================================
# SELFIES DECODER  (same as main pipeline — rebuilt here)
# We rebuild vocab from data; in production load from checkpoint.
# ============================================================

class SelfiesDecoder(nn.Module):
    def __init__(self, vocab_size: int, stoi: dict, itos: dict):
        super().__init__()
        self.stoi = stoi
        self.itos = itos
        PAD = stoi["[PAD]"]
        self.pad_idx = PAD
        self.sos_idx = stoi["[SOS]"]
        self.eos_idx = stoi["[EOS]"]

        self.token_emb = nn.Embedding(vocab_size, LATENT_DIM, padding_idx=PAD)
        self.pos_emb   = nn.Embedding(MAX_LEN + 4, LATENT_DIM)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=LATENT_DIM, nhead=8,
            dim_feedforward=LATENT_DIM * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=6)
        self.norm    = nn.LayerNorm(LATENT_DIM)
        self.fc      = nn.Linear(LATENT_DIM, vocab_size)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, z: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        B, T  = tgt.shape
        pos   = torch.arange(T, device=tgt.device).unsqueeze(0)
        tgt_emb  = self.token_emb(tgt) + self.pos_emb(pos)
        memory   = z.unsqueeze(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)
        tgt_pad  = tgt == self.pad_idx
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_pad)
        return self.fc(self.norm(out))

    @torch.no_grad()
    def greedy_decode(self, z: torch.Tensor, max_len: int = MAX_LEN) -> list[str]:
        self.eval()
        B      = z.size(0)
        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=z.device)
        for _ in range(max_len):
            logits     = self.forward(z, tokens)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens     = torch.cat([tokens, next_token], dim=1)
        results = []
        for row in tokens[:, 1:]:
            toks = []
            for t in row.tolist():
                if t == self.eos_idx:
                    break
                if t not in (self.pad_idx, self.sos_idx):
                    toks.append(self.itos.get(t, ""))
            results.append("".join(toks))
        return results

# ============================================================
# CATALYST TRANSFORMER  (the trainable middle layer)
# ============================================================

class CatalystTransformer(nn.Module):
    """
    Maps a variable-length set of input molecule embeddings
    (reactants, solvents, products) to MAX_CATALYSTS catalyst latent vectors.

    Architecture:
      1. Linear projection: LATENT_DIM → CATGEN_D_MODEL  (per-molecule)
      2. Transformer Encoder to contextualise the input set
      3. Learned catalyst query tokens (MAX_CATALYSTS of them)
      4. Cross-attention Transformer Decoder: queries attend to context
      5. Linear projection back: CATGEN_D_MODEL → LATENT_DIM  (per catalyst)
    """

    def __init__(self, max_input_mols: int = 20):
        super().__init__()
        self.max_input_mols = max_input_mols

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(LATENT_DIM, CATGEN_D_MODEL),
            nn.LayerNorm(CATGEN_D_MODEL),
        )

        # Positional embedding for input set (learned)
        self.input_pos_emb = nn.Embedding(max_input_mols + 1, CATGEN_D_MODEL)

        # Transformer encoder: contextualises the input molecule set
        enc_layer = nn.TransformerEncoderLayer(
            d_model=CATGEN_D_MODEL, nhead=CATGEN_NHEAD,
            dim_feedforward=CATGEN_FF_DIM,
            dropout=CATGEN_DROPOUT, batch_first=True, norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(enc_layer, num_layers=CATGEN_LAYERS // 2)

        # Learned catalyst query tokens
        self.catalyst_queries = nn.Parameter(
            torch.randn(MAX_CATALYSTS, CATGEN_D_MODEL) * 0.02
        )

        # Transformer decoder: catalyst queries attend to context
        dec_layer = nn.TransformerDecoderLayer(
            d_model=CATGEN_D_MODEL, nhead=CATGEN_NHEAD,
            dim_feedforward=CATGEN_FF_DIM,
            dropout=CATGEN_DROPOUT, batch_first=True, norm_first=True,
        )
        self.catalyst_decoder = nn.TransformerDecoder(dec_layer, num_layers=CATGEN_LAYERS)

        # Output projection back to LATENT_DIM (for SELFIES decoder)
        self.output_proj = nn.Sequential(
            nn.Linear(CATGEN_D_MODEL, LATENT_DIM * 2),
            nn.GELU(),
            nn.Dropout(CATGEN_DROPOUT),
            nn.Linear(LATENT_DIM * 2, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
        )

        # Confidence head: predicts how many catalysts are present (0 → MAX_CATALYSTS)
        self.confidence_head = nn.Linear(CATGEN_D_MODEL, 1)

    def forward(self, mol_embeddings: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        mol_embeddings : (B, N, LATENT_DIM)  — N ≤ max_input_mols
        padding_mask   : (B, N) bool mask, True = padding position

        Returns
        -------
        catalyst_latents : (B, MAX_CATALYSTS, LATENT_DIM)
        confidence       : (B, MAX_CATALYSTS)  — logit for each catalyst slot being "active"
        """
        B, N, _ = mol_embeddings.shape

        # Project + add positional embeddings
        pos  = torch.arange(N, device=mol_embeddings.device).unsqueeze(0)  # (1, N)
        ctx  = self.input_proj(mol_embeddings) + self.input_pos_emb(pos)    # (B, N, D)

        # Encode the input context
        ctx = self.context_encoder(ctx, src_key_padding_mask=padding_mask)  # (B, N, D)

        # Expand catalyst queries for batch
        queries = self.catalyst_queries.unsqueeze(0).expand(B, -1, -1)     # (B, K, D)

        # Cross-attend: catalyst queries attend to context
        cat_out = self.catalyst_decoder(queries, ctx,
                                        memory_key_padding_mask=padding_mask)  # (B, K, D)

        # Confidence per catalyst slot
        confidence = self.confidence_head(cat_out).squeeze(-1)              # (B, K)

        # Project to LATENT_DIM
        catalyst_latents = self.output_proj(cat_out)                        # (B, K, LATENT_DIM)

        return catalyst_latents, confidence

# ============================================================
# FULL CATALYST GENERATION MODEL
# ============================================================

class CatalystModel(nn.Module):
    """
    Wraps GINEncoder + CatalystTransformer + SelfiesDecoder.
    During Phase 1: gin and decoder are frozen.
    During Phase 2: all parameters are trainable.
    """

    def __init__(self, gin: GINEncoder, transformer: CatalystTransformer,
                 decoder: SelfiesDecoder):
        super().__init__()
        self.gin         = gin
        self.transformer = transformer
        self.decoder     = decoder

    def encode_smiles_list(self, smiles_list: list[str]) -> torch.Tensor:
        """
        Encode a flat list of SMILES → (len, LATENT_DIM).
        Uses GIN for graph encoding (or ChemBERTa as fallback).
        """
        embeddings = []
        for sm in smiles_list:
            g = smiles_to_graph(sm)
            if g is not None:
                batch_g = Batch.from_data_list([g]).to(DEVICE)
                emb = self.gin(batch_g)                        # (1, LATENT_DIM)
            else:
                emb = embed_smiles_batch([sm])                 # ChemBERTa fallback
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0)                    # (N, LATENT_DIM)

    def forward(self,
                input_mol_embeddings: torch.Tensor,            # (B, N, LATENT_DIM)
                padding_mask: torch.Tensor | None,
                catalyst_seqs: torch.Tensor,                   # (B, K, T) teacher-forced
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits     : (B, K, T, VOCAB_SIZE)
        confidence : (B, K)
        """
        cat_latents, confidence = self.transformer(input_mol_embeddings, padding_mask)
        B, K, _ = cat_latents.shape

        # Decode each catalyst slot with teacher forcing
        # Reshape (B*K, LATENT_DIM) and (B*K, T) for batch processing
        z       = cat_latents.view(B * K, LATENT_DIM)
        seqs    = catalyst_seqs.view(B * K, -1)                # (B*K, T)

        inp     = seqs[:, :-1]
        logits  = self.decoder(z, inp)                         # (B*K, T-1, VOCAB_SIZE)
        logits  = logits.view(B, K, seqs.size(1) - 1, -1)

        return logits, confidence

    @torch.no_grad()
    def predict(self, input_smiles: list[str],
                conf_threshold: float = 0.0) -> list[str]:
        """
        Given a flat list of input SMILES (reactants / solvents / products),
        predict catalyst SELFIES strings.
        conf_threshold: only return catalyst slots with logit > threshold.
        """
        self.gin.eval()
        self.transformer.eval()
        self.decoder.eval()

        embs        = self.encode_smiles_list(input_smiles)    # (N, LATENT_DIM)
        embs        = embs.unsqueeze(0)                        # (1, N, LATENT_DIM)
        cat_latents, confidence = self.transformer(embs, None)

        # confidence: (1, K)
        active = (confidence[0] > conf_threshold).tolist()

        z       = cat_latents[0]                               # (K, LATENT_DIM)
        selfies_list = self.decoder.greedy_decode(z)

        results = []
        for i, (s, a) in enumerate(zip(selfies_list, active)):
            if a:
                try:
                    smiles = sf.decoder(s)
                    if smiles and Chem.MolFromSmiles(smiles):
                        results.append(smiles)
                except Exception:
                    pass
        return results

# ============================================================
# DATASET  (one for each model variant)
# ============================================================

def parse_smiles_str(s: str) -> list[str]:
    """Split a dot-separated SMILES string into individual molecules."""
    if not isinstance(s, str):
        return []
    return [m.strip() for m in s.split(".") if m.strip()]

def clean_list(x):
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else []
    except Exception:
        return []

def build_vocab(molecules: list[str]):
    """Build SELFIES vocabulary from a list of SMILES."""
    valid_selfies = []
    smiles_to_selfies = {}
    for sm in tqdm(molecules, desc="SMILES → SELFIES"):
        try:
            s = sf.encoder(sm)
            valid_selfies.append(s)
            smiles_to_selfies[sm] = s
        except Exception:
            continue
    alphabet = sorted(sf.get_alphabet_from_selfies(valid_selfies))
    SPECIAL  = ["[PAD]", "[SOS]", "[EOS]"]
    vocab    = SPECIAL + alphabet
    stoi     = {t: i for i, t in enumerate(vocab)}
    itos     = {i: t for t, i in stoi.items()}
    return stoi, itos, smiles_to_selfies

def encode_selfies_seq(smiles: str, stoi: dict,
                       smiles_to_selfies: dict) -> torch.Tensor:
    s      = smiles_to_selfies.get(smiles)
    if s is None:
        return torch.full((MAX_LEN,), stoi["[PAD]"], dtype=torch.long)
    tokens = list(sf.split_selfies(s))
    ids    = [stoi["[SOS]"]] + [stoi.get(t, stoi["[PAD]"]) for t in tokens] + [stoi["[EOS]"]]
    ids    = ids[:MAX_LEN]
    ids   += [stoi["[PAD]"]] * (MAX_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class CatalystDataset(Dataset):
    """
    mode: "A" = reactants + solvents
          "B" = reactants + products
          "C" = reactants + products + solvents
    """

    def __init__(self, df: pd.DataFrame, mode: str,
                 stoi: dict, smiles_to_selfies: dict,
                 max_input_mols: int = 20):
        assert mode in ("A", "B", "C")
        self.mode             = mode
        self.stoi             = stoi
        self.smiles_to_selfies = smiles_to_selfies
        self.max_input_mols   = max_input_mols
        self.rows             = []

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"Building dataset (Mode {mode})"):
            reactants = parse_smiles_str(row.get("reactants", ""))
            products  = parse_smiles_str(row.get("products", ""))
            solvents  = row.get("solvents", [])
            if not isinstance(solvents, list):
                solvents = []
            catalysts = row.get("catalyst", [])
            if not isinstance(catalysts, list):
                catalysts = []

            # Filter to valid SMILES only
            reactants = [s for s in reactants if Chem.MolFromSmiles(s)]
            products  = [s for s in products  if Chem.MolFromSmiles(s)]
            solvents  = [s for s in solvents  if Chem.MolFromSmiles(s)]
            catalysts = [s for s in catalysts if Chem.MolFromSmiles(s)
                         and s in smiles_to_selfies]

            if not reactants or not catalysts:
                continue

            if mode == "A":
                input_mols = reactants + solvents
            elif mode == "B":
                input_mols = reactants + products
            else:  # C
                input_mols = reactants + products + solvents

            input_mols = [m for m in input_mols if Chem.MolFromSmiles(m)]
            if not input_mols:
                continue

            # Pad / truncate catalysts to MAX_CATALYSTS
            cats_padded = catalysts[:MAX_CATALYSTS]
            cats_padded += [""] * (MAX_CATALYSTS - len(cats_padded))

            self.rows.append({
                "input_mols": input_mols,
                "catalysts":  cats_padded,
                "n_cats":     len(catalysts[:MAX_CATALYSTS]),
            })

        print(f"Dataset Mode {mode}: {len(self.rows)} samples")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row        = self.rows[idx]
        input_mols = row["input_mols"]
        catalysts  = row["catalysts"]
        n_cats     = row["n_cats"]

        # Pre-compute ChemBERTa embeddings for input molecules
        # (we do this at dataset level to keep training fast)
        with torch.no_grad():
            embs = embed_smiles_batch(input_mols)              # (N, 768)

        N = embs.size(0)
        # Pad to max_input_mols
        if N < self.max_input_mols:
            pad = torch.zeros(
    self.max_input_mols - N,
    LATENT_DIM,
    device=embs.device
)
            embs = torch.cat([embs, pad], dim=0)
        else:
            embs = embs[:self.max_input_mols]
            N    = self.max_input_mols

        pad_mask = torch.zeros(self.max_input_mols, dtype=torch.bool)
        pad_mask[N:] = True

        # Encode catalyst SELFIES sequences
        cat_seqs = []
        for sm in catalysts:
            if sm:
                seq = encode_selfies_seq(sm, self.stoi, self.smiles_to_selfies)
            else:
                seq = torch.full((MAX_LEN,), self.stoi["[PAD]"], dtype=torch.long)
            cat_seqs.append(seq)
        cat_seqs = torch.stack(cat_seqs)                       # (K, T)

        # Confidence targets: 1 for active catalyst slots, 0 for padding
        conf_target = torch.zeros(MAX_CATALYSTS)
        conf_target[:n_cats] = 1.0

        return embs, pad_mask, cat_seqs, conf_target


def make_loader(dataset, shuffle=True):
    return DataLoader(dataset, batch_size=BATCH_SIZE,
                      shuffle=shuffle, drop_last=True,
                      num_workers=0, pin_memory=False)

# ============================================================
# TRAINING UTILITIES
# ============================================================

def set_frozen(model: CatalystModel, freeze_gin_decoder: bool):
    """Freeze or unfreeze GIN and Decoder."""
    for p in model.gin.parameters():
        p.requires_grad = not freeze_gin_decoder
    for p in model.decoder.parameters():
        p.requires_grad = not freeze_gin_decoder
    for p in model.transformer.parameters():
        p.requires_grad = True    # transformer always trains

def make_optimizer(model: CatalystModel, freeze_gin_decoder: bool, lr: float):
    if freeze_gin_decoder:
        return torch.optim.AdamW(model.transformer.parameters(),
                                 lr=lr, weight_decay=1e-4)
    else:
        return torch.optim.AdamW([
            {"params": model.transformer.parameters(), "lr": lr},
            {"params": model.gin.parameters(),         "lr": lr * 0.1},
            {"params": model.decoder.parameters(),     "lr": lr * 0.1},
        ], weight_decay=1e-4)


def train_one_epoch(model: CatalystModel, loader: DataLoader,
                    optimizer, criterion_seq, criterion_conf,
                    epoch: int, total_epochs: int,
                    model_name: str, vocab_size: int) -> dict:
    model.train()
    total_loss = total_seq = total_conf = total_acc = 0.0
    pbar = tqdm(loader, desc=f"[{model_name}] Epoch {epoch}/{total_epochs}")

    PAD_IDX = list(model.decoder.stoi.values())[0]  # fallback
    PAD_IDX = model.decoder.pad_idx

    for embs, pad_mask, cat_seqs, conf_target in pbar:
        embs        = embs.to(DEVICE)          # (B, N, LATENT_DIM)
        pad_mask    = pad_mask.to(DEVICE)      # (B, N)
        cat_seqs    = cat_seqs.to(DEVICE)      # (B, K, T)
        conf_target = conf_target.to(DEVICE)   # (B, K)

        # Forward
        logits, confidence = model(embs, pad_mask, cat_seqs)
        # logits: (B, K, T-1, VOCAB_SIZE)
        # confidence: (B, K)

        B, K, T_minus1, V = logits.shape

        # Sequence loss: cross-entropy over active catalyst slots only
        # target is cat_seqs[:, :, 1:] (shifted)
        tgt = cat_seqs[:, :, 1:]               # (B, K, T-1)

        # Mask inactive slots (conf_target == 0)
        active_mask = conf_target > 0.5        # (B, K)

        # Flatten
        logits_flat = logits.reshape(B * K, T_minus1, V)
        tgt_flat    = tgt.reshape(B * K, T_minus1)
        active_flat = active_mask.reshape(B * K)

        # Only compute reconstruction loss on active slots
        if active_flat.any():
            logits_active = logits_flat[active_flat]           # (n_active, T-1, V)
            tgt_active    = tgt_flat[active_flat]              # (n_active, T-1)
            seq_loss = criterion_seq(
                logits_active.reshape(-1, V),
                tgt_active.reshape(-1),
            )
        else:
            seq_loss = torch.tensor(0.0, device=DEVICE)

        # Confidence loss: BCE
        conf_loss = criterion_conf(confidence, conf_target)

        loss = seq_loss + 0.5 * conf_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        # Token accuracy (active slots, non-PAD)
        with torch.no_grad():
            if active_flat.any():
                preds  = logits_active.argmax(-1)              # (n_active, T-1)
                mask   = tgt_active != PAD_IDX
                acc    = ((preds == tgt_active) & mask).float().sum() / mask.float().sum().clamp(min=1)
            else:
                acc = torch.tensor(0.0)

        total_loss += loss.item()
        total_seq  += seq_loss.item()
        total_conf += conf_loss.item()
        total_acc  += acc.item()

        pbar.set_postfix({
            "loss":     f"{loss.item():.4f}",
            "seq":      f"{seq_loss.item():.4f}",
            "conf":     f"{conf_loss.item():.4f}",
            "tok_acc":  f"{acc.item():.3f}",
        })

    n = len(loader)
    return {
        "loss":     total_loss / n,
        "seq_loss": total_seq  / n,
        "conf_loss":total_conf / n,
        "tok_acc":  total_acc  / n,
    }


def save_checkpoint(model: CatalystModel, stoi: dict, epoch: int,
                    phase: int, metrics: dict, save_dir: str, model_name: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir,
                        f"{model_name}_phase{phase}_epoch{epoch:02d}.pt")
    torch.save({
        "epoch":                epoch,
        "phase":                phase,
        "model_name":           model_name,
        "gin_state_dict":       model.gin.state_dict(),
        "transformer_state_dict": model.transformer.state_dict(),
        "decoder_state_dict":   model.decoder.state_dict(),
        "vocab":                stoi,
        "metrics":              metrics,
        "config": {
            "LATENT_DIM":      LATENT_DIM,
            "GIN_HIDDEN":      GIN_HIDDEN,
            "GIN_LAYERS":      GIN_LAYERS,
            "ATOM_FEATS":      ATOM_FEATS,
            "CATGEN_D_MODEL":  CATGEN_D_MODEL,
            "CATGEN_NHEAD":    CATGEN_NHEAD,
            "CATGEN_LAYERS":   CATGEN_LAYERS,
            "MAX_CATALYSTS":   MAX_CATALYSTS,
            "MAX_LEN":         MAX_LEN,
        },
    }, path)
    print(f"  Checkpoint saved → {path}")


# ============================================================
# LOAD CHECKPOINT UTILITY
# ============================================================

def load_catalyst_model(checkpoint_path: str,
                        stoi: dict, itos: dict,
                        device: torch.device) -> CatalystModel:
    """Load a full CatalystModel from a saved checkpoint."""
    ckpt    = torch.load(checkpoint_path, map_location=device)
    vocab_size = len(ckpt.get("vocab", stoi))
    # Use vocab from checkpoint if available
    saved_stoi = ckpt.get("vocab", stoi)
    saved_itos = {i: t for t, i in saved_stoi.items()}

    gin         = GINEncoder().to(device)
    transformer = CatalystTransformer().to(device)
    decoder     = SelfiesDecoder(vocab_size, saved_stoi, saved_itos).to(device)

    gin.load_state_dict(ckpt["gin_state_dict"])
    transformer.load_state_dict(ckpt["transformer_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])

    model = CatalystModel(gin, transformer, decoder).to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}  (epoch {ckpt['epoch']}, phase {ckpt['phase']})")
    return model


# ============================================================
# TRAIN ONE CATALYST MODEL (both phases)
# ============================================================

def train_catalyst_model(model_name: str, mode: str,
                         df: pd.DataFrame, stoi: dict, itos: dict,
                         smiles_to_selfies: dict,
                         gin_checkpoint: str | None = None,
                         decoder_checkpoint: str | None = None):
    """
    Full training pipeline for one catalyst model variant.

    model_name : "A", "B", or "C"
    mode       : same as model_name, controls input composition
    """
    print(f"\n{'='*60}")
    print(f"  CATALYST MODEL {model_name}  (mode={mode})")
    print(f"{'='*60}")

    save_dir = os.path.join(SAVE_DIR_BASE, f"model_{model_name}")
    os.makedirs(save_dir, exist_ok=True)

    vocab_size = len(stoi)

    # Build dataset + loader
    dataset = CatalystDataset(df, mode, stoi, smiles_to_selfies)
    loader  = make_loader(dataset)

    # Instantiate components
    gin         = GINEncoder().to(DEVICE)
    transformer = CatalystTransformer().to(DEVICE)
    decoder     = SelfiesDecoder(vocab_size, stoi, itos).to(DEVICE)

    # Load pretrained GIN weights
    if gin_checkpoint and os.path.exists(gin_checkpoint):
        ckpt = torch.load(gin_checkpoint, map_location=DEVICE)
        key  = "model_state_dict" if "model_state_dict" in ckpt else \
               "gin_state_dict"   if "gin_state_dict"   in ckpt else None
        if key:
            gin.load_state_dict(ckpt[key])
            print(f"  GIN loaded from {gin_checkpoint} ✓")
        else:
            gin.load_state_dict(ckpt)

    # Load pretrained Decoder weights
    if decoder_checkpoint and os.path.exists(decoder_checkpoint):
        ckpt = torch.load(decoder_checkpoint, map_location=DEVICE)
        key  = "model_state_dict" if "model_state_dict" in ckpt else \
               "decoder_state_dict" if "decoder_state_dict" in ckpt else None
        if key:
            # Need to reconcile vocab size — skip if mismatch
            try:
                decoder.load_state_dict(ckpt[key], strict=False)
                print(f"  Decoder loaded from {decoder_checkpoint} ✓")
            except Exception as e:
                print(f"  Decoder load warning: {e}")
        else:
            try:
                decoder.load_state_dict(ckpt, strict=False)
            except Exception as e:
                print(f"  Decoder load warning: {e}")

    model = CatalystModel(gin, transformer, decoder).to(DEVICE)

    criterion_seq  = nn.CrossEntropyLoss(ignore_index=stoi["[PAD]"],
                                         label_smoothing=0.1)
    criterion_conf = nn.BCEWithLogitsLoss()

    # ── Phase 1: freeze GIN + Decoder, train transformer only ────────────
    print(f"\n  Phase 1 — CatalystTransformer only ({PHASE1_EPOCHS} epochs)")
    set_frozen(model, freeze_gin_decoder=True)
    optimizer = make_optimizer(model, freeze_gin_decoder=True, lr=LR_PHASE1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PHASE1_EPOCHS
    )

    for epoch in range(1, PHASE1_EPOCHS + 1):
        metrics = train_one_epoch(
            model, loader, optimizer, criterion_seq, criterion_conf,
            epoch, PHASE1_EPOCHS, model_name, vocab_size,
        )
        scheduler.step()
        print(f"  [P1 E{epoch}] loss={metrics['loss']:.4f}  "
              f"seq={metrics['seq_loss']:.4f}  "
              f"conf={metrics['conf_loss']:.4f}  "
              f"tok_acc={metrics['tok_acc']:.4f}")
        save_checkpoint(model, stoi, epoch, phase=1,
                        metrics=metrics, save_dir=save_dir,
                        model_name=f"catalyst_{model_name}")

    # ── Phase 2: full finetuning ──────────────────────────────────────────
    print(f"\n  Phase 2 — Full finetune GIN + Transformer + Decoder ({PHASE2_EPOCHS} epochs)")
    set_frozen(model, freeze_gin_decoder=False)
    optimizer = make_optimizer(model, freeze_gin_decoder=False, lr=LR_PHASE2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PHASE2_EPOCHS
    )

    for epoch in range(1, PHASE2_EPOCHS + 1):
        metrics = train_one_epoch(
            model, loader, optimizer, criterion_seq, criterion_conf,
            epoch, PHASE2_EPOCHS, model_name, vocab_size,
        )
        scheduler.step()
        print(f"  [P2 E{epoch}] loss={metrics['loss']:.4f}  "
              f"seq={metrics['seq_loss']:.4f}  "
              f"conf={metrics['conf_loss']:.4f}  "
              f"tok_acc={metrics['tok_acc']:.4f}")
        save_checkpoint(model, stoi, epoch, phase=2,
                        metrics=metrics, save_dir=save_dir,
                        model_name=f"catalyst_{model_name}")

    print(f"\n  Model {model_name} training complete ✓")
    return model


# ============================================================
# QUICK VALIDATION
# ============================================================

@torch.no_grad()
def validate_model(model: CatalystModel, df: pd.DataFrame,
                   mode: str, n_samples: int = 20):
    """
    Run a quick validity check on n_samples reactions.
    Reports how many predicted catalysts are valid SMILES.
    """
    model.gin.eval()
    model.transformer.eval()
    model.decoder.eval()

    samples = df.sample(min(n_samples, len(df))).iterrows()
    valid_total = 0
    total_pred  = 0

    print(f"\n  Validation (Mode {mode}, {n_samples} samples)")
    for _, row in samples:
        reactants = parse_smiles_str(row.get("reactants", ""))
        products  = parse_smiles_str(row.get("products", ""))
        solvents  = row.get("solvents", []) or []
        true_cats = row.get("catalyst", []) or []

        reactants = [s for s in reactants if Chem.MolFromSmiles(s)]
        products  = [s for s in products  if Chem.MolFromSmiles(s)]
        solvents  = [s for s in solvents  if Chem.MolFromSmiles(s)]

        if mode == "A":
            input_mols = reactants + solvents
        elif mode == "B":
            input_mols = reactants + products
        else:
            input_mols = reactants + products + solvents

        if not input_mols:
            continue

        predicted = model.predict(input_mols, conf_threshold=0.0)
        total_pred  += len(predicted)
        valid_total += len(predicted)  # predict() already checks validity

        print(f"    true: {true_cats[:2]}  |  predicted: {predicted[:2]}")

    print(f"  Predicted {total_pred} valid catalysts over {n_samples} reactions.")

