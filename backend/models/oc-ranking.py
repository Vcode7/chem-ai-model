"""
OC20-Style Catalyst Energy Scoring & Ranking Model
====================================================

ARCHITECTURE OVERVIEW
---------------------
Input: A single "system" = adsorbate (reactant) + catalyst surface
       Represented as: atomic coordinates, atom types, unit cell box

Stage 1 — Structural Encoder (SchNet-style GNN)
  - Builds a radius graph from 3D coordinates
  - Message passing with edge features = radial basis of interatomic distances
  - Per-atom embeddings then summed → system-level embedding

Stage 2 — Transformer Interaction Layer
  - Cross-attends over all atom tokens
  - Captures long-range adsorbate–surface interactions

Stage 3 — Energy Head
  - MLP → scalar predicted energy (eV)

Training Task: Predict DFT adsorption energy (regression, MAE loss)
Catalyst Ranking: At inference, score multiple catalysts + same reactant,
                  sort ascending by predicted energy → best catalyst ranked #1

DATA FORMAT (DeePMD/OC20 numpy format per sys directory)
---------------------------------------------------------
  coord.npy       : (N_frames, N_atoms * 3)  fractional or Cartesian coords
  energy.npy      : (N_frames,)              DFT total/adsorption energy
  force.npy       : (N_frames, N_atoms * 3)  optional atomic forces
  box.npy         : (N_frames, 9)            lattice vectors (row-major 3x3)
  type.raw        : (N_atoms,)               integer atom type indices
  type_map.raw    : list of element symbols  (maps index → element)
  real_atom_numbs.npy  : (N_frames, N_types) atoms of each type per frame
  real_atom_types.npy  : (N_frames, N_atoms) integer type for each atom
"""

import os
import glob
import math
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paths — adjust to your directory layout
TRAIN_ROOT = "/kaggle/input/datasets/vikaskag7/oc2m-dataset/OC2M/train"          # directory with DeePMD-style system folders
VALID_ROOT = "/kaggle/input/datasets/vikaskag7/oc2m-dataset/OC2M/valid"
CHECKPOINT_DIR = "./checkpoints_catalyst"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# From input.json type_map
TYPE_MAP = [
    "Ag","Al","As","Au","B","Bi","C","Ca","Cd","Cl","Co","Cr","Cs","Cu","Fe",
    "Ga","Ge","H","Hf","Hg","In","Ir","K","Mg","Mn","Mo","N","Na","Nb","Ni",
    "O","Os","P","Pb","Pd","Pt","Rb","Re","Rh","Ru","S","Sb","Sc","Se","Si",
    "Sn","Sr","Ta","Tc","Te","Ti","Tl","V","W","Y","Zn","Zr",
]
N_ELEM = len(TYPE_MAP)          # 57

# Model hyper-params
EMBED_DIM    = 128              # atom embedding dimension
HIDDEN_DIM   = 256
N_ATTN_HEADS = 8
N_TRANSFORMER_LAYERS = 3
N_RBF        = 64               # radial basis functions
RCUT         = 9.0              # Angstrom, from input.json rcut
MAX_ATOMS    = 256              # pad sequences to this length

# Training hyper-params
BATCH_SIZE   = 8
LR           = 3e-4
N_EPOCHS     = 50
WEIGHT_DECAY = 1e-4
FORCE_COEFF  = 0.1              # weight of force loss (set 0 to disable)


# ─────────────────────────────────────────
# RADIAL BASIS FUNCTIONS
# ─────────────────────────────────────────

class GaussianRBF(nn.Module):
    """Gaussian radial basis functions over [0, rcut]."""

    def __init__(self, n_rbf: int = N_RBF, rcut: float = RCUT):
        super().__init__()
        centers = torch.linspace(0.0, rcut, n_rbf)
        self.register_buffer("centers", centers)
        self.width = (rcut / n_rbf) ** 2

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """r: (...,) → (..., n_rbf)"""
        return torch.exp(-((r.unsqueeze(-1) - self.centers) ** 2) / self.width)


# ─────────────────────────────────────────
# GNN MESSAGE PASSING BLOCK
# ─────────────────────────────────────────

class InteractionBlock(nn.Module):
    """
    SchNet-style continuous filter convolution:
      h_i ← h_i + aggr_{j in N(i)} (W * rbf(r_ij)) @ h_j
    """

    def __init__(self, embed_dim: int, n_rbf: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h: torch.Tensor,           # (B, N, D)
        rbf: torch.Tensor,         # (B, N, N, n_rbf) — pairwise RBF
        mask: torch.Tensor,        # (B, N) bool — True for real atoms
    ) -> torch.Tensor:
        B, N, D = h.shape
        # Filter weights: (B, N, N, D)
        W = self.filter_net(rbf)
        # Zero out padding pairs
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, N, N)
        W = W * pair_mask.unsqueeze(-1).float()
        # Message = W * h_j: (B, N, N, D)
        msg = W * h.unsqueeze(1)          # broadcast h_j over i dim
        agg = msg.sum(dim=2)              # (B, N, D)
        return self.norm(h + self.out_proj(agg))


# ─────────────────────────────────────────
# STRUCTURAL ENCODER (GNN BACKBONE)
# ─────────────────────────────────────────

class StructureEncoder(nn.Module):
    """
    Encodes atom coordinates + types into per-atom embeddings,
    then aggregates to a system-level vector.
    """

    def __init__(
        self,
        n_elem: int   = N_ELEM,
        embed_dim: int = EMBED_DIM,
        n_rbf: int    = N_RBF,
        n_layers: int = 3,
        rcut: float   = RCUT,
    ):
        super().__init__()
        self.rcut = rcut
        self.elem_emb = nn.Embedding(n_elem + 1, embed_dim, padding_idx=n_elem)
        self.rbf = GaussianRBF(n_rbf, rcut)
        self.interactions = nn.ModuleList(
            [InteractionBlock(embed_dim, n_rbf) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        atom_types: torch.Tensor,    # (B, N) int — padded with n_elem
        coords: torch.Tensor,        # (B, N, 3) float
        mask: torch.Tensor,          # (B, N) bool — True = real atom
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h         : (B, N, D) per-atom embeddings
          system_emb: (B, D)    sum-pooled system embedding
        """
        # Atom embeddings
        h = self.elem_emb(atom_types)  # (B, N, D)

        # Pairwise distances
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)   # (B, N, N, 3)
        dist = diff.norm(dim=-1)                            # (B, N, N)
        # Mask self-loops and beyond cutoff
        within = (dist < self.rcut) & (dist > 1e-6)
        within = within & mask.unsqueeze(2) & mask.unsqueeze(1)

        # RBF encoding, zeroed outside cutoff
        rbf = self.rbf(dist) * within.unsqueeze(-1).float()  # (B,N,N,n_rbf)

        for layer in self.interactions:
            h = layer(h, rbf, mask)

        h = self.out_norm(h)

        # Sum pool over real atoms
        h_masked = h * mask.unsqueeze(-1).float()
        system_emb = h_masked.sum(dim=1)  # (B, D)

        return h, system_emb


# ─────────────────────────────────────────
# TRANSFORMER INTERACTION LAYER
# ─────────────────────────────────────────

class AtomTransformer(nn.Module):
    """Standard pre-norm transformer encoder applied to per-atom tokens."""

    def __init__(self, embed_dim: int, n_heads: int, n_layers: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        h: torch.Tensor,       # (B, N, D)
        mask: torch.Tensor,    # (B, N) bool — True = real (NOT padding)
    ) -> torch.Tensor:         # (B, N, D)
        # TransformerEncoder expects padding mask where True = IGNORE
        src_key_padding_mask = ~mask  # True for padding positions
        return self.encoder(h, src_key_padding_mask=src_key_padding_mask)


# ─────────────────────────────────────────
# ENERGY & FORCE HEAD
# ─────────────────────────────────────────

class EnergyForceHead(nn.Module):
    """
    MLP that maps the system-level embedding → predicted energy.
    Forces are obtained as -∇_{coords} E via autograd.
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, system_emb: torch.Tensor) -> torch.Tensor:
        """system_emb: (B, D) → predicted_energy: (B,)"""
        return self.mlp(system_emb).squeeze(-1)


# ─────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────

class CatalystEnergyModel(nn.Module):
    """
    Full pipeline:
      atom_types + coords → GNN → Transformer → Energy scalar
    """

    def __init__(
        self,
        n_elem: int         = N_ELEM,
        embed_dim: int      = EMBED_DIM,
        hidden_dim: int     = HIDDEN_DIM,
        n_rbf: int          = N_RBF,
        n_gnn_layers: int   = 3,
        n_attn_heads: int   = N_ATTN_HEADS,
        n_attn_layers: int  = N_TRANSFORMER_LAYERS,
        rcut: float         = RCUT,
    ):
        super().__init__()
        self.encoder = StructureEncoder(n_elem, embed_dim, n_rbf, n_gnn_layers, rcut)
        self.transformer = AtomTransformer(embed_dim, n_attn_heads, n_attn_layers)
        # Project from atom-level back to system-level after transformer
        self.pool_proj = nn.Linear(embed_dim, embed_dim)
        self.head = EnergyForceHead(embed_dim, hidden_dim)

    def forward(
        self,
        atom_types: torch.Tensor,   # (B, N)
        coords: torch.Tensor,       # (B, N, 3)
        mask: torch.Tensor,         # (B, N) bool
    ) -> torch.Tensor:
        # GNN encoding
        h, _ = self.encoder(atom_types, coords, mask)
        # Transformer refinement
        h = self.transformer(h, mask)
        # Pool to system level
        h_masked = h * mask.unsqueeze(-1).float()
        n_atoms = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        system_emb = h_masked.sum(dim=1) / n_atoms     # mean pool (B, D)
        system_emb = self.pool_proj(system_emb)
        # Predict energy
        energy = self.head(system_emb)
        return energy


# ─────────────────────────────────────────
# DATA LOADING (DeePMD format)
# ─────────────────────────────────────────
def load_system(sys_dir: str):

    sys_dir = Path(sys_dir)

    set_dir = sys_dir / "set.000"

    required = [
        set_dir / "coord.npy",
        set_dir / "energy.npy",
        set_dir / "real_atom_types.npy",
    ]

    for f in required:
        if not f.exists():
            return None

    try:

        coord = np.load(
            set_dir / "coord.npy"
        ).astype(np.float32)

        energy = np.load(
            set_dir / "energy.npy"
        ).astype(np.float32)

        real_atom_types = np.load(
            set_dir / "real_atom_types.npy"
        ).astype(np.int64)

        force = None
        if (set_dir / "force.npy").exists():
            force = np.load(
                set_dir / "force.npy"
            ).astype(np.float32)

        box = None
        if (set_dir / "box.npy").exists():
            box = np.load(
                set_dir / "box.npy"
            ).astype(np.float32)

        n_frames = coord.shape[0]
        n_atoms = real_atom_types.shape[1]

        if coord.shape[1] != n_atoms * 3:
            print(f"Coord mismatch in {sys_dir}")
            return None

        return {
            "coord": coord,
            "energy": energy,
            "types": real_atom_types,
            "force": force,
            "box": box,
            "n_atoms": n_atoms,
            "n_frames": n_frames,
            "sys_dir": str(sys_dir),
        }

    except Exception as e:
        print(f"Failed loading {sys_dir}")
        print(e)
        return None

class OC20Dataset(Dataset):
    """
    Flat dataset of (frame, system) pairs loaded from DeePMD-format directories.
    Each item is one MD snapshot with its energy (and optional forces).
    """

    def __init__(self, sys_dirs: List[str], max_atoms: int = MAX_ATOMS):
        self.max_atoms = max_atoms
        self.samples = []   # list of (types, coord_frame, energy, force_frame)

        n_loaded = 0
        for d in sys_dirs:
            data = load_system(d)
            if data is None:
                continue
            if data["n_atoms"] > max_atoms:
                continue   # skip oversized systems
            for i in range(data["n_frames"]):
                self.samples.append({
                    "types":  data["types"][i],                          # (N,)
                    "coord":  data["coord"][i].reshape(-1, 3),         # (N, 3)
                    "energy": float(data["energy"][i]),
                    "force":  data["force"][i].reshape(-1, 3) if data["force"] is not None else None,
                })
            n_loaded += 1

        print(f"  Loaded {len(self.samples)} frames from {n_loaded} systems.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        n = len(s["types"])
        pad = self.max_atoms - n

        # Atom types (padded with n_elem = 57)
        types_padded = np.concatenate([s["types"], np.full(pad, N_ELEM, dtype=int)])

        # Coords (padded with zeros)
        coord_padded = np.concatenate([s["coord"], np.zeros((pad, 3), dtype=np.float32)])

        # Mask
        mask = np.array([True]*n + [False]*pad)

        return {
            "atom_types": torch.tensor(types_padded, dtype=torch.long),
            "coords":     torch.tensor(coord_padded, dtype=torch.float32),
            "mask":       torch.tensor(mask,         dtype=torch.bool),
            "energy":     torch.tensor(s["energy"],  dtype=torch.float32),
        }


def discover_systems(root: str) -> List[str]:
    """
    Find system directories containing:
        type.raw
        set.000/
    """

    pattern = os.path.join(root, "**", "type.raw")

    paths = glob.glob(pattern, recursive=True)

    # return parent directory of type.raw
    sys_dirs = [os.path.dirname(p) for p in paths]

    return sys_dirs


# ─────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────

def build_loaders(train_root: str, valid_root: str, batch_size: int):
    print("Discovering training systems...")
    train_dirs = discover_systems(train_root)
    print(f"  Found {len(train_dirs)} system directories in {train_root}")
    print("Building training dataset...")
    train_ds = OC20Dataset(train_dirs)

    print("Discovering validation systems...")
    valid_dirs = discover_systems(valid_root)
    print(f"  Found {len(valid_dirs)} system directories in {valid_root}")
    print("Building validation dataset...")
    valid_ds = OC20Dataset(valid_dirs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, valid_loader


def energy_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_mae  = 0.0
    n_batches  = 0

    for batch in loader:
        atom_types = batch["atom_types"].to(device)
        coords     = batch["coords"].to(device)
        mask       = batch["mask"].to(device)
        energy_gt  = batch["energy"].to(device)

        pred_energy = model(atom_types, coords, mask)
        loss = energy_mae(pred_energy, energy_gt)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_mae  += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1), total_mae / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mae = 0.0
    n_batches = 0
    for batch in loader:
        atom_types = batch["atom_types"].to(device)
        coords     = batch["coords"].to(device)
        mask       = batch["mask"].to(device)
        energy_gt  = batch["energy"].to(device)
        pred_energy = model(atom_types, coords, mask)
        total_mae += energy_mae(pred_energy, energy_gt).item()
        n_batches += 1
    return total_mae / max(n_batches, 1)


def save_checkpoint(model, optimizer, epoch, val_mae, path):
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "val_mae":     val_mae,
        "config": {
            "n_elem": N_ELEM, "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
            "n_rbf": N_RBF, "rcut": RCUT, "n_attn_heads": N_ATTN_HEADS,
            "n_attn_layers": N_TRANSFORMER_LAYERS,
        }
    }, path)
    print(f"  Saved checkpoint → {path}")


# ─────────────────────────────────────────
# CATALYST RANKING INFERENCE
# ─────────────────────────────────────────

def rank_catalysts(
    model: CatalystEnergyModel,
    reactant_types: np.ndarray,         # (N_ads,) atom types of adsorbate
    reactant_coords: np.ndarray,        # (N_ads, 3)
    catalyst_systems: List[dict],       # list of {"types", "coords", "name"}
    device: torch.device,
    max_atoms: int = MAX_ATOMS,
) -> List[dict]:
    """
    Given a fixed adsorbate (reactant) and a list of candidate catalyst surfaces,
    combine each catalyst+adsorbate, predict the adsorption energy, and rank
    catalysts by ascending predicted energy (lower = thermodynamically favored).

    Returns sorted list of dicts with keys: name, predicted_energy, rank.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for cat in catalyst_systems:
            # Concatenate adsorbate atoms + catalyst surface atoms
            combined_types  = np.concatenate([reactant_types,  cat["types"]])
            combined_coords = np.concatenate([reactant_coords, cat["coords"]])
            n = len(combined_types)

            if n > max_atoms:
                print(f"  Warning: {cat['name']} has {n} atoms > MAX_ATOMS={max_atoms}, skipping.")
                continue

            pad = max_atoms - n
            types_padded = np.concatenate([combined_types, np.full(pad, N_ELEM, dtype=int)])
            coord_padded = np.concatenate([combined_coords, np.zeros((pad, 3), dtype=np.float32)])
            mask = np.array([True]*n + [False]*pad)

            atom_types_t = torch.tensor(types_padded, dtype=torch.long).unsqueeze(0).to(device)
            coords_t     = torch.tensor(coord_padded, dtype=torch.float32).unsqueeze(0).to(device)
            mask_t       = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)

            pred_e = model(atom_types_t, coords_t, mask_t).item()
            results.append({"name": cat["name"], "predicted_energy_eV": pred_e})

    # Sort ascending — lower adsorption energy = stronger binding = better catalyst
    results.sort(key=lambda x: x["predicted_energy_eV"])
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    return results


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    # ── Build data loaders ──────────────────────────────────────────────
    train_loader, valid_loader = build_loaders(TRAIN_ROOT, VALID_ROOT, BATCH_SIZE)

    if len(train_loader) == 0:
        raise RuntimeError(
            f"No training data found under {TRAIN_ROOT}. "
            "Check that coord.npy / energy.npy / type.raw exist in subdirectories."
        )

    # ── Build model ─────────────────────────────────────────────────────
    model = CatalystEnergyModel().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    best_val_mae = float("inf")

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\nTraining for {N_EPOCHS} epochs on {DEVICE}...\n")
    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, None, DEVICE)
        val_mae = evaluate(model, valid_loader, DEVICE) if len(valid_loader) > 0 else float("nan")
        scheduler.step()

        print(f"Epoch {epoch:3d}/{N_EPOCHS} | "
              f"train MAE: {train_mae:.4f} eV | val MAE: {val_mae:.4f} eV")

        # Save periodic checkpoint
        if epoch % 5 == 0 or epoch == 1:
            save_checkpoint(
                model, optimizer, epoch, val_mae,
                os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch:03d}.pt")
            )

        # Save best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(
                model, optimizer, epoch, val_mae,
                os.path.join(CHECKPOINT_DIR, "model_best.pt")
            )

    print(f"\nTraining complete. Best val MAE: {best_val_mae:.4f} eV")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}/")

    # ── Example ranking inference ─────────────────────────────────────────
    # NOTE: Replace this with real system data after training.
    print("\n── Example Catalyst Ranking (random dummy data) ──")

    # Dummy adsorbate: CO molecule (2 atoms)
    reactant_types  = np.array([6, 30], dtype=int)   # C=index 6, O=index 30
    reactant_coords = np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.2]], dtype=np.float32)

    # Dummy catalysts: 3 example surfaces
    dummy_catalysts = []
    for i, name in enumerate(["Pt(111)", "Cu(111)", "Ni(111)"]):
        n_surf = 12
        types  = np.random.randint(0, N_ELEM, size=n_surf)
        coords = np.random.randn(n_surf, 3).astype(np.float32) * 2.0
        coords[:, 2] -= 3.0  # shift below adsorbate
        dummy_catalysts.append({"name": name, "types": types, "coords": coords})

    rankings = rank_catalysts(model, reactant_types, reactant_coords,
                               dummy_catalysts, DEVICE)

    print("\nCatalyst Rankings (lower predicted energy = better):")
    print(f"{'Rank':<6} {'Catalyst':<15} {'Pred. Energy (eV)':>18}")
    print("-" * 42)
    for r in rankings:
        print(f"{r['rank']:<6} {r['name']:<15} {r['predicted_energy_eV']:>18.4f}")


if __name__ == "__main__":
    main()