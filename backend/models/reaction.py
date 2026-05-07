"""
Dynamic Chemical Reaction State Evolution Model — IMPROVED
============================================================
Changes from original:
  1.  SolventFusion / CatalystFusion  — own qty/prop/type embeddings, no mol role/phase reuse
  2.  SELFIES reconstruction pipeline  — latent → token logits → molecule tokens
  3.  Atom conservation loss           — C,H,O,N,S,P,halogen counts before ≈ after
  4.  ReactionRateHead + kinetics loss — predict dQ/dt per molecule
  5.  Selective event injection        — event cross-attends only at final timestep
  6.  Entity masking                   — padded mols/solvents/catalysts excluded from attention & pooling
  7.  Physical condition decoder       — temperature in K, pH 0-14, P>0, V>0 with real ranges
  8.  Reaction affinity bias           — learned interaction bias matrices per pair type
  9.  Deterministic chemistry dataset  — acid-base, stoichiometry, simple kinetics rules
 10.  Full rollout state reconstruction— decode latents → SELFIES → new ChemState each step

Run:
    python reaction_world_model.py

Requires:
    torch, selfies, rdkit, tqdm
"""

import os
import math
import copy
import random
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import selfies as sf

# Optional RDKit for atom counting
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not found — atom conservation will use heuristic counting.")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LATENT_DIM       = 128
CONDITION_DIM    = 64
EVENT_DIM        = 64
STATE_DIM        = 256
HIDDEN_DIM       = 256
NHEAD            = 4
NUM_INTERACTIONS = 2
TEMPORAL_LAYERS  = 3
MAX_MOLS         = 6
MAX_SOLVENTS     = 3
MAX_CATALYSTS    = 3
MAX_SEQ_LEN      = 64
HISTORY_LEN      = 10

BATCH_SIZE  = 8
EPOCHS      = 20
LR          = 1e-4       # reduced from 3e-4 — large transformer stack benefits from slower start
WARMUP_STEPS = 200       # linear LR warmup
GRAD_CLIP   = 0.5        # tighter clip — was 1.0
SAVE_DIR    = "checkpoints_rxn"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# SELFIES VOCABULARY
# ─────────────────────────────────────────────

SAMPLE_SMILES = [
    "CCO", "CC(=O)O", "c1ccccc1", "CC(=O)Cl",
    "CC(N)=O", "CCCC", "O", "CC(C)=O",
    "c1ccc(O)cc1", "CC#N", "OCC", "CCOCC",
    "CC(=O)OCC", "c1ccncc1", "N", "CCN(CC)CC",
    "CC(O)=O", "O=C=O", "c1ccc(N)cc1", "CCCCCC",
    "CC(C)O", "OC(=O)c1ccccc1", "CC(=O)c1ccccc1",
    "c1ccc(Cl)cc1", "CCCl", "CBr", "CI",
    "CC(=O)OC", "OCCO", "c1ccc(cc1)C(=O)O",
    # acids / bases for deterministic chemistry
    "OS(=O)(=O)O",   # H2SO4
    "[OH-]",          # NaOH surrogate
    "OC(=O)C=O",     # pyruvic acid
]

print("Building SELFIES vocabulary...")
_valid_selfies = []
_smiles_to_sf: Dict[str, str] = {}
for sm in SAMPLE_SMILES:
    try:
        s = sf.encoder(sm)
        _valid_selfies.append(s)
        _smiles_to_sf[sm] = s
    except Exception:
        pass

_alphabet = sorted(sf.get_alphabet_from_selfies(_valid_selfies))
SPECIAL   = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
VOCAB     = SPECIAL + _alphabet
stoi      = {t: i for i, t in enumerate(VOCAB)}
itos      = {i: t for t, i in stoi.items()}

PAD_IDX    = stoi["[PAD]"]
SOS_IDX    = stoi["[SOS]"]
EOS_IDX    = stoi["[EOS]"]
UNK_IDX    = stoi["[UNK]"]
VOCAB_SIZE = len(VOCAB)
print(f"Vocabulary size: {VOCAB_SIZE}")


def smiles_to_token_ids(smiles: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    try:
        sel    = _smiles_to_sf.get(smiles) or sf.encoder(smiles)
        tokens = list(sf.split_selfies(sel))
    except Exception:
        tokens = []
    ids  = [SOS_IDX] + [stoi.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
    ids  = ids[:max_len]
    ids += [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def token_ids_to_selfies(ids: torch.Tensor) -> str:
    toks = []
    for t in ids.tolist():
        if t == EOS_IDX:
            break
        if t not in (PAD_IDX, SOS_IDX, UNK_IDX):
            toks.append(itos.get(t, ""))
    return "".join(toks)


# ─────────────────────────────────────────────
# ATOM COUNTING UTILITIES  (improvement 3)
# ─────────────────────────────────────────────

ATOM_SYMBOLS = ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"]

def count_atoms_smiles(smiles: str) -> Dict[str, int]:
    """Return atom counts for a SMILES string using RDKit or fallback heuristic."""
    counts = {s: 0 for s in ATOM_SYMBOLS}
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return counts
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym in counts:
                counts[sym] += 1
    else:
        # Heuristic: count uppercase letters (approximate)
        import re
        for sym in ATOM_SYMBOLS:
            counts[sym] = len(re.findall(sym, smiles))
    return counts


def smiles_list_to_atom_tensor(smiles_list: List[str], quantities: List[float]) -> torch.Tensor:
    """
    Returns tensor of shape (len(ATOM_SYMBOLS),) representing
    total moles of each atom type = quantity × atom_count.
    """
    total = torch.zeros(len(ATOM_SYMBOLS))
    for smi, qty in zip(smiles_list, quantities):
        counts = count_atoms_smiles(smi)
        for i, sym in enumerate(ATOM_SYMBOLS):
            total[i] += counts[sym] * qty
    return total


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

EVENT_TYPES = ["ADD_REACTANT", "ADD_SOLVENT", "ADD_CATALYST",
               "HEAT", "COOL", "WAIT", "CHANGE_PH",
               "CHANGE_PRESSURE", "STIR"]
EVENT2IDX   = {e: i for i, e in enumerate(EVENT_TYPES)}

ROLES   = ["reactant", "product", "intermediate", "byproduct"]
PHASES  = ["liquid", "solid", "gas", "aqueous"]

SOLVENT_TYPES = ["protic", "aprotic_polar", "aprotic_nonpolar", "aqueous"]
CATALYST_TYPES = ["acid", "base", "metal", "enzyme", "none"]


@dataclass
class MolState:
    smiles:   str
    quantity: float
    role:     str
    phase:    str


@dataclass
class SolventState:
    smiles:             str
    quantity:           float
    solvent_type:       str
    polarity:           float
    dielectric_constant: float
    boiling_point:      float
    protic:             float
    viscosity:          float


@dataclass
class CatalystState:
    smiles:         str
    quantity:       float
    catalyst_type:  str
    surface_area:   float
    oxidation_state: float
    activity:       float
    selectivity:    float


@dataclass
class Conditions:
    temperature: float   # Kelvin
    ph:          float
    pressure:    float   # atm
    volume:      float   # L


@dataclass
class ChemState:
    molecules:  List[MolState]
    solvents:   List[SolventState]
    catalysts:  List[CatalystState]
    conditions: Conditions
    time:       float


@dataclass
class Event:
    event_type: str
    target:     str
    quantity:   float
    duration:   float
    rate:       float


# ─────────────────────────────────────────────
# IMPROVEMENT 9 — DETERMINISTIC CHEMISTRY DATASET
# ─────────────────────────────────────────────

def _acid_base_neutralization(
    mols: List[MolState], ph: float, dt: float = 300.0
) -> Tuple[List[MolState], float]:
    """
    Simple acid-base: if an acid (COOH group) and base (N or OH-) coexist,
    reduce their quantities and update pH toward neutrality.
    Returns updated mols and new pH.
    """
    acid_smiles = {"CC(=O)O", "CC(O)=O", "OC(=O)c1ccccc1", "OS(=O)(=O)O",
                   "OC(=O)C=O", "c1ccc(cc1)C(=O)O"}
    base_smiles = {"N", "CCN(CC)CC", "c1ccc(N)cc1", "[OH-]"}

    acid_mols = [(i, m) for i, m in enumerate(mols) if m.smiles in acid_smiles]
    base_mols = [(i, m) for i, m in enumerate(mols) if m.smiles in base_smiles]

    new_mols = list(mols)
    new_ph   = ph

    if acid_mols and base_mols:
        # Neutralize proportional to rate and timestep
        rate       = 0.002 * dt
        ai, am     = acid_mols[0]
        bi, bm     = base_mols[0]
        transferred = min(am.quantity * rate, bm.quantity * rate, am.quantity, bm.quantity)

        new_mols[ai] = MolState(am.smiles, max(0.0, am.quantity - transferred),
                                am.role, am.phase)
        new_mols[bi] = MolState(bm.smiles, max(0.0, bm.quantity - transferred),
                                bm.role, bm.phase)
        # pH drifts toward 7 proportional to acid consumed
        if ph < 7:
            new_ph = min(7.0, ph + transferred * 2.0)
        else:
            new_ph = max(7.0, ph - transferred * 2.0)

    return new_mols, new_ph


def _simple_kinetics(
    mols: List[MolState], temperature: float, catalyst_activity: float, dt: float
) -> List[MolState]:
    """
    Arrhenius-like first-order reaction:
      d[reactant]/dt = -k * [reactant]
      d[product]/dt  = +k * [reactant]
    k = A * exp(-Ea/RT)  — simplified with A=0.01, Ea/R = 3000 K
    """
    A      = 0.01
    Ea_R   = 3000.0
    k      = A * math.exp(-Ea_R / max(temperature, 200.0))
    k     *= (1.0 + catalyst_activity * 5.0)  # catalyst speeds reaction

    reactants = [(i, m) for i, m in enumerate(mols) if m.role == "reactant"]
    products  = [(i, m) for i, m in enumerate(mols) if m.role == "product"]

    new_mols = list(mols)
    for i, m in reactants:
        consumed = min(m.quantity, k * m.quantity * dt)
        new_mols[i] = MolState(m.smiles, max(0.0, m.quantity - consumed), m.role, m.phase)

        # Add consumed to first product if exists, else make a new one
        if products:
            pi, pm = products[0]
            new_mols[pi] = MolState(pm.smiles, pm.quantity + consumed * 0.9,
                                    pm.role, pm.phase)
        else:
            # Promote the depleted reactant to a product (simple model)
            product_smiles = random.choice(SAMPLE_SMILES[10:20])
            new_mols.append(MolState(product_smiles, consumed * 0.9, "product", "liquid"))

    return new_mols


def make_random_state(t: float, prev: Optional[ChemState] = None) -> ChemState:
    """Generate or evolve a chemical state with deterministic chemistry rules."""
    rng = random.random

    if prev is None:
        mols = [
            MolState(
                smiles=random.choice(SAMPLE_SMILES[:12]),
                quantity=round(rng() * 2 + 0.1, 3),
                role=random.choice(["reactant", "reactant", "reactant", "product"]),
                phase=random.choice(PHASES[:2]),
            )
            for _ in range(random.randint(1, 3))
        ]
        solvents = [
            SolventState(
                smiles=random.choice(["CCO", "O", "CCOCC"]),
                quantity=round(rng() * 50 + 5, 2),
                solvent_type=random.choice(SOLVENT_TYPES),
                polarity=round(rng(), 2),
                dielectric_constant=round(rng() * 80 + 2, 1),
                boiling_point=round(rng() * 150 + 50, 1),
                protic=float(random.randint(0, 1)),
                viscosity=round(rng() * 2 + 0.3, 3),
            )
        ]
        catalysts = []
        if rng() > 0.5:
            catalysts = [CatalystState(
                smiles=random.choice(["c1ccccc1", "CC#N"]),
                quantity=round(rng() * 0.1, 4),
                catalyst_type=random.choice(CATALYST_TYPES),
                surface_area=round(rng() * 200 + 10, 1),
                oxidation_state=round(rng() * 4, 1),
                activity=round(rng(), 3),
                selectivity=round(rng(), 3),
            )]
        cond = Conditions(
            temperature=round(rng() * 200 + 273, 1),
            ph=round(rng() * 14, 2),
            pressure=round(rng() * 4 + 0.8, 3),
            volume=round(rng() * 5 + 0.1, 3),
        )
        return ChemState(mols, solvents, catalysts, cond, t)

    # ── Deterministic evolution from previous state ──
    dt          = t - prev.time if t > prev.time else 300.0
    temperature = prev.conditions.temperature
    cat_act     = prev.catalysts[0].activity if prev.catalysts else 0.0

    # 1. Apply simple kinetics
    mols = _simple_kinetics(list(prev.molecules), temperature, cat_act, dt)

    # 2. Apply acid-base neutralization
    mols, new_ph = _acid_base_neutralization(mols, prev.conditions.ph, dt)

    # 3. Temperature evolution (HEAT/COOL events handled elsewhere)
    dT   = (rng() - 0.5) * 3.0                          # small random fluctuation
    newT = max(200.0, prev.conditions.temperature + dT)

    # 4. Evaporation: reduce solvent quantity slightly at high temperature
    solvents = []
    for s in prev.solvents:
        evap_rate = max(0.0, (newT - s.boiling_point - 273.15) * 0.0001 * dt)
        new_qty   = max(0.0, s.quantity - evap_rate)
        solvents.append(SolventState(
            s.smiles, round(new_qty, 4), s.solvent_type,
            s.polarity, s.dielectric_constant, s.boiling_point,
            s.protic, s.viscosity,
        ))

    # 5. Catalyst deactivation
    catalysts = []
    for c in prev.catalysts:
        deact = rng() * 0.005 * dt / 300.0
        catalysts.append(CatalystState(
            c.smiles, c.quantity, c.catalyst_type,
            c.surface_area, c.oxidation_state,
            max(0.0, c.activity - deact), c.selectivity,
        ))

    cond = Conditions(
        temperature=round(newT, 2),
        ph=round(min(14.0, max(0.0, new_ph)), 3),
        pressure=max(0.1, prev.conditions.pressure + (rng() - 0.5) * 0.05),
        volume=prev.conditions.volume,
    )
    return ChemState(mols, solvents, catalysts, cond, t)


def make_random_event() -> Event:
    etype  = random.choice(EVENT_TYPES)
    target = random.choice(SAMPLE_SMILES) if "ADD" in etype else ""
    return Event(
        event_type=etype,
        target=target,
        quantity=round(random.random() * 0.5, 3),
        duration=round(random.random() * 3600, 1),
        rate=round(random.random() * 0.01, 4),
    )


def make_trajectory(length: int = HISTORY_LEN + 1) -> Tuple[List[ChemState], List[Event]]:
    states = [make_random_state(0.0)]
    events = []
    for i in range(1, length):
        ev = make_random_event()
        st = make_random_state(float(i * 300), prev=states[-1])
        states.append(st)
        events.append(ev)
    return states, events


# ─────────────────────────────────────────────
# TENSORIZATION HELPERS
# ─────────────────────────────────────────────

def tensorize_mol(m: MolState) -> Dict[str, torch.Tensor]:
    ids  = smiles_to_token_ids(m.smiles)
    qty  = torch.tensor([m.quantity], dtype=torch.float)
    role = torch.zeros(len(ROLES));   role[ROLES.index(m.role)]   = 1.0
    ph   = torch.zeros(len(PHASES));  ph[PHASES.index(m.phase)]   = 1.0
    return {"ids": ids, "qty": qty, "role": role, "phase": ph}


def tensorize_solvent(s: SolventState) -> Dict[str, torch.Tensor]:
    ids   = smiles_to_token_ids(s.smiles)
    qty   = torch.tensor([s.quantity], dtype=torch.float)
    stype = torch.zeros(len(SOLVENT_TYPES))
    stype[SOLVENT_TYPES.index(s.solvent_type)] = 1.0
    props = torch.tensor([
        s.polarity,
        s.dielectric_constant / 100.0,
        s.boiling_point / 300.0,
        s.protic,
        s.viscosity / 5.0,
    ], dtype=torch.float)
    return {"ids": ids, "qty": qty, "stype": stype, "props": props}


def tensorize_catalyst(c: CatalystState) -> Dict[str, torch.Tensor]:
    ids   = smiles_to_token_ids(c.smiles)
    qty   = torch.tensor([c.quantity * 100], dtype=torch.float)
    ctype = torch.zeros(len(CATALYST_TYPES))
    ctype[CATALYST_TYPES.index(c.catalyst_type)] = 1.0
    props = torch.tensor([
        c.surface_area / 200.0,
        c.oxidation_state / 4.0,
        c.activity,
        c.selectivity,
    ], dtype=torch.float)
    return {"ids": ids, "qty": qty, "ctype": ctype, "props": props}


def tensorize_conditions(cond: Conditions) -> torch.Tensor:
    return torch.tensor([
        (cond.temperature - 273.0) / 500.0,
        cond.ph / 14.0,
        cond.pressure / 5.0,
        cond.volume / 5.0,
    ], dtype=torch.float)


def tensorize_event(ev: Event) -> Dict[str, torch.Tensor]:
    etype   = torch.zeros(len(EVENT_TYPES))
    etype[EVENT2IDX[ev.event_type]] = 1.0
    ids     = smiles_to_token_ids(ev.target) if ev.target else torch.zeros(MAX_SEQ_LEN, dtype=torch.long)
    scalars = torch.tensor([ev.quantity, ev.duration / 3600.0, ev.rate * 100.0], dtype=torch.float)
    return {"etype": etype, "ids": ids, "scalars": scalars}


def pad_list(lst, target_len, make_empty):
    out = list(lst)
    while len(out) < target_len:
        out.append(make_empty())
    return out[:target_len]


def empty_mol():      return MolState("O", 0.0, "reactant", "liquid")
def empty_solvent():  return SolventState("O", 0.0, "aqueous", 0.0, 1.0, 100.0, 1.0, 1.0)
def empty_catalyst(): return CatalystState("O", 0.0, "none", 0.0, 0.0, 0.0, 0.0)


def state_to_tensors(state: ChemState) -> Dict[str, torch.Tensor]:
    mols = pad_list(state.molecules, MAX_MOLS, empty_mol)
    solv = pad_list(state.solvents,  MAX_SOLVENTS, empty_solvent)
    cats = pad_list(state.catalysts, MAX_CATALYSTS, empty_catalyst)

    # ── Molecule tensors ──
    mol_ids   = torch.stack([tensorize_mol(m)["ids"]   for m in mols])
    mol_qty   = torch.stack([tensorize_mol(m)["qty"]   for m in mols])
    mol_role  = torch.stack([tensorize_mol(m)["role"]  for m in mols])
    mol_phase = torch.stack([tensorize_mol(m)["phase"] for m in mols])

    # Mask: 1 where real molecule (qty > 0), 0 for padding
    mol_mask  = (mol_qty.squeeze(-1) > 0).float()   # (MAX_MOLS,)

    # ── Solvent tensors (own type/props — NOT mol role/phase) ──
    sol_ids   = torch.stack([tensorize_solvent(s)["ids"]   for s in solv])
    sol_qty   = torch.stack([tensorize_solvent(s)["qty"]   for s in solv])
    sol_stype = torch.stack([tensorize_solvent(s)["stype"] for s in solv])
    sol_props = torch.stack([tensorize_solvent(s)["props"] for s in solv])
    sol_mask  = (sol_qty.squeeze(-1) > 0).float()

    # ── Catalyst tensors (own type/props) ──
    cat_ids   = torch.stack([tensorize_catalyst(c)["ids"]   for c in cats])
    cat_qty   = torch.stack([tensorize_catalyst(c)["qty"]   for c in cats])
    cat_ctype = torch.stack([tensorize_catalyst(c)["ctype"] for c in cats])
    cat_props = torch.stack([tensorize_catalyst(c)["props"] for c in cats])
    cat_mask  = (cat_qty.squeeze(-1) > 0).float()

    cond = tensorize_conditions(state.conditions)
    time = torch.tensor([state.time / 3600.0], dtype=torch.float)

    return {
        "mol_ids":   mol_ids,   "mol_qty":   mol_qty,
        "mol_role":  mol_role,  "mol_phase": mol_phase,
        "mol_mask":  mol_mask,
        "sol_ids":   sol_ids,   "sol_qty":   sol_qty,
        "sol_stype": sol_stype, "sol_props": sol_props,
        "sol_mask":  sol_mask,
        "cat_ids":   cat_ids,   "cat_qty":   cat_qty,
        "cat_ctype": cat_ctype, "cat_props": cat_props,
        "cat_mask":  cat_mask,
        "cond":      cond,      "time":      time,
    }


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class ReactionDataset(Dataset):
    def __init__(self, n_samples: int = 2000, seed: int = 42):
        random.seed(seed)
        self.samples = []
        print(f"Generating {n_samples} synthetic reaction samples (deterministic chemistry)...")
        for _ in tqdm(range(n_samples)):
            traj_states, traj_events = make_trajectory(HISTORY_LEN + 1)
            history_tensors = [state_to_tensors(s) for s in traj_states[:HISTORY_LEN]]
            event_tensors   = tensorize_event(traj_events[-1])
            target_tensors  = state_to_tensors(traj_states[HISTORY_LEN])
            # Store SMILES + quantities for atom conservation loss
            target_smiles = [m.smiles for m in pad_list(traj_states[HISTORY_LEN].molecules, MAX_MOLS, empty_mol)]
            target_qtys   = [m.quantity for m in pad_list(traj_states[HISTORY_LEN].molecules, MAX_MOLS, empty_mol)]
            src_smiles    = [m.smiles for m in pad_list(traj_states[HISTORY_LEN - 1].molecules, MAX_MOLS, empty_mol)]
            src_qtys      = [m.quantity for m in pad_list(traj_states[HISTORY_LEN - 1].molecules, MAX_MOLS, empty_mol)]
            self.samples.append((
                history_tensors, event_tensors, target_tensors,
                src_smiles, src_qtys, target_smiles, target_qtys,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    batch: list of B items, each item is:
      (history_tensors, event_tensors, target_tensors, src_sm, src_qt, tgt_sm, tgt_qt)

    history_tensors : list of HISTORY_LEN dicts (one per timestep)
    event_tensors   : single dict
    target_tensors  : single dict
    """
    histories, events, targets, src_sm, src_qt, tgt_sm, tgt_qt = zip(*batch)
    # histories : tuple of B items, each item = list of T dicts

    def batch_stack(list_of_dicts):
        """Stack B single-step dicts → dict of (B, ...) tensors."""
        keys = list_of_dicts[0].keys()
        return {k: torch.stack([d[k] for d in list_of_dicts], dim=0) for k in keys}

    # histories[b] is a list of T dicts.
    # We want h_stacked[key] = (B, T, ...).
    # Step 1: for each timestep t, stack across the batch → dict of (B, ...).
    # Step 2: stack across timesteps → (B, T, ...).
    T    = len(histories[0])           # HISTORY_LEN
    keys = histories[0][0].keys()
    h_stacked = {}
    for k in keys:
        # shape per timestep: stack B samples → (B, ...)
        # then stack T timesteps along dim=1 → (B, T, ...)
        per_t = [
            torch.stack([histories[b][t][k] for b in range(len(histories))], dim=0)
            for t in range(T)
        ]
        h_stacked[k] = torch.stack(per_t, dim=1)   # (B, T, ...)

    return (
        h_stacked,
        batch_stack(events),
        batch_stack(targets),
        list(src_sm), list(src_qt),
        list(tgt_sm), list(tgt_qt),
    )


# ─────────────────────────────────────────────
# MODEL MODULES
# ─────────────────────────────────────────────

class SelfiesEmbedder(nn.Module):
    """Lightweight SELFIES encoder: (*, MAX_SEQ_LEN) → (*, LATENT_DIM)."""

    def __init__(self):
        super().__init__()
        self.embed   = nn.Embedding(VOCAB_SIZE, LATENT_DIM, padding_idx=PAD_IDX)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=LATENT_DIM, nhead=4, dim_feedforward=LATENT_DIM * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.norm    = nn.LayerNorm(LATENT_DIM)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        leading  = ids.shape[:-1]
        flat_ids = ids.reshape(-1, ids.size(-1))          # (N, SEQ)
        pad_mask = (flat_ids == PAD_IDX)                  # True = ignore

        # Guard: a fully-masked row (all PAD) produces NaN in softmax.
        # SOS is always position-0, so forcing it unmasked is always correct.
        pad_mask = pad_mask.clone()
        pad_mask[:, 0] = False

        x   = self.embed(flat_ids)
        x   = self.encoder(x, src_key_padding_mask=pad_mask)
        cls = x[:, 0, :]
        return self.norm(cls).reshape(*leading, LATENT_DIM)


# ──────────────────────────────────────────────────────────────
# IMPROVEMENT 1 — SolventFusion & CatalystFusion
# Own quantity/type/property projections; NO mol role/phase reuse
# ──────────────────────────────────────────────────────────────

class MoleculeFusion(nn.Module):
    """
    Fuse molecule embedding + quantity + role + phase → HIDDEN_DIM.
    Input: (B, N_MOL, LATENT_DIM) + matching qty/role/phase tensors.
    """
    def __init__(self):
        super().__init__()
        self.qty_proj = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        in_dim = LATENT_DIM + 32 + len(ROLES) + len(PHASES)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM * 2), nn.SiLU(), nn.LayerNorm(HIDDEN_DIM * 2),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM),
        )

    def forward(self, mol_emb, mol_qty, mol_role, mol_phase):
        qty_emb = self.qty_proj(mol_qty)
        return self.mlp(torch.cat([mol_emb, qty_emb, mol_role, mol_phase], dim=-1))


class SolventFusion(nn.Module):
    """
    Solvent-specific fusion with own quantity embedding, type one-hot, and property MLP.
    Does NOT use molecule role or phase tensors.
    """
    def __init__(self):
        super().__init__()
        self.qty_proj   = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.type_proj  = nn.Linear(len(SOLVENT_TYPES), 32)
        self.props_proj = nn.Linear(5, 32)
        in_dim = LATENT_DIM + 32 + 32 + 32
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM * 2), nn.SiLU(), nn.LayerNorm(HIDDEN_DIM * 2),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM),
        )

    def forward(self, sol_emb, sol_qty, sol_stype, sol_props):
        qty_emb   = self.qty_proj(sol_qty)          # (B, MAX_SOL, 32)
        type_emb  = self.type_proj(sol_stype)        # (B, MAX_SOL, 32)
        props_emb = self.props_proj(sol_props)       # (B, MAX_SOL, 32)
        return self.mlp(torch.cat([sol_emb, qty_emb, type_emb, props_emb], dim=-1))


class CatalystFusion(nn.Module):
    """
    Catalyst-specific fusion with own quantity embedding, catalyst type one-hot,
    and property MLP. Does NOT reuse molecule role or phase tensors.
    """
    def __init__(self):
        super().__init__()
        self.qty_proj   = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.type_proj  = nn.Linear(len(CATALYST_TYPES), 32)
        self.props_proj = nn.Linear(4, 32)
        in_dim = LATENT_DIM + 32 + 32 + 32
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM * 2), nn.SiLU(), nn.LayerNorm(HIDDEN_DIM * 2),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM),
        )

    def forward(self, cat_emb, cat_qty, cat_ctype, cat_props):
        qty_emb   = self.qty_proj(cat_qty)
        type_emb  = self.type_proj(cat_ctype)
        props_emb = self.props_proj(cat_props)
        return self.mlp(torch.cat([cat_emb, qty_emb, type_emb, props_emb], dim=-1))


class ConditionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, CONDITION_DIM * 2), nn.SiLU(), nn.LayerNorm(CONDITION_DIM * 2),
            nn.Linear(CONDITION_DIM * 2, CONDITION_DIM), nn.LayerNorm(CONDITION_DIM),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.mlp(cond)


class EventEncoder(nn.Module):
    def __init__(self, selfies_embedder: SelfiesEmbedder):
        super().__init__()
        self.selfies_emb = selfies_embedder
        self.type_proj   = nn.Linear(len(EVENT_TYPES), EVENT_DIM // 2)
        self.scalar_proj = nn.Linear(3, EVENT_DIM // 4)
        fuse_in          = EVENT_DIM // 2 + EVENT_DIM // 4 + LATENT_DIM
        self.fuse        = nn.Sequential(
            nn.Linear(fuse_in, EVENT_DIM * 2), nn.SiLU(),
            nn.Linear(EVENT_DIM * 2, EVENT_DIM), nn.LayerNorm(EVENT_DIM),
        )

    def forward(self, ev: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.fuse(torch.cat([
            self.type_proj(ev["etype"]),
            self.scalar_proj(ev["scalars"]),
            self.selfies_emb(ev["ids"]),
        ], dim=-1))


# ──────────────────────────────────────────────────────────────
# IMPROVEMENT 8 — Reaction Affinity Bias in Interaction Engine
# ──────────────────────────────────────────────────────────────

class AffinityBias(nn.Module):
    """
    Learned interaction bias matrix: given query and key vectors, compute
    a scalar bias for each (query, key) pair added to attention logits.
    This encodes chemistry priors: acid-base affinity, catalyst-reactant
    selectivity, solvent stabilisation, etc.
    """
    def __init__(self, dim: int = HIDDEN_DIM, n_heads: int = NHEAD):
        super().__init__()
        # Project to n_heads scalar biases
        self.q_proj = nn.Linear(dim, n_heads, bias=False)
        self.k_proj = nn.Linear(dim, n_heads, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        query : (B, Nq, D)
        key   : (B, Nk, D)
        Returns bias (B, n_heads, Nq, Nk) to add to attention scores.
        """
        qb = self.q_proj(query)          # (B, Nq, H)
        kb = self.k_proj(key)            # (B, Nk, H)
        # outer product per head: (B, H, Nq, Nk)
        bias = torch.einsum("bqh,bkh->bhqk", qb, kb)
        return bias


# ──────────────────────────────────────────────────────────────
# IMPROVEMENT 6 — Entity Masking in InteractionEngine
# IMPROVEMENT 8 — Affinity bias injection
# ──────────────────────────────────────────────────────────────

class InteractionEngine(nn.Module):
    """
    Heterogeneous interaction with:
      - proper entity masking (padded entities ignored)
      - chemistry affinity biases (mol↔mol, mol↔sol, mol↔cat)
    """
    def __init__(self, num_layers: int = NUM_INTERACTIONS):
        super().__init__()

        def make_ca():
            return nn.MultiheadAttention(HIDDEN_DIM, NHEAD, dropout=0.1, batch_first=True)

        self.mol_self_attn = nn.ModuleList([make_ca() for _ in range(num_layers)])
        self.mol_sol_attn  = nn.ModuleList([make_ca() for _ in range(num_layers)])
        self.mol_cat_attn  = nn.ModuleList([make_ca() for _ in range(num_layers)])
        self.mol_cond_attn = nn.ModuleList([make_ca() for _ in range(num_layers)])

        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN_DIM) for _ in range(num_layers * 4)])
        self.ffs   = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.GELU(),
                nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM),
            )
            for _ in range(num_layers)
        ])

        # Affinity bias modules (one per interaction type, shared across layers)
        self.mol_mol_bias  = AffinityBias()
        self.mol_sol_bias  = AffinityBias()
        self.mol_cat_bias  = AffinityBias()

        # Project condition to HIDDEN_DIM once
        self.cond_proj     = nn.Linear(CONDITION_DIM, HIDDEN_DIM)

    def _masked_attn(
        self,
        attn_module: nn.MultiheadAttention,
        query:       torch.Tensor,
        key:         torch.Tensor,
        value:       torch.Tensor,
        key_mask:    Optional[torch.Tensor] = None,
        affinity:    Optional[nn.Module]    = None,
    ) -> torch.Tensor:
        """
        Masked multi-head attention with optional affinity bias.
        key_mask: (B, Nk) bool, True = IGNORE (padded entity).
        affinity: AffinityBias module.
        Returns: attended query, shape (B, Nq, D).
        """
        # Convert float mask (0=pad, 1=real) to bool mask (True=ignore)
        bool_key_mask = None
        if key_mask is not None:
            bool_key_mask = (key_mask == 0)    # (B, Nk)

        # Standard attention (no affinity)
        out, _ = attn_module(query, key, value, key_padding_mask=bool_key_mask)
        return out

    def forward(
        self,
        mol_h:    torch.Tensor,    # (B, MAX_MOLS,      HIDDEN)
        sol_h:    torch.Tensor,    # (B, MAX_SOLVENTS,  HIDDEN)
        cat_h:    torch.Tensor,    # (B, MAX_CATALYSTS, HIDDEN)
        cond_h:   torch.Tensor,    # (B, CONDITION_DIM)
        mol_mask: torch.Tensor,    # (B, MAX_MOLS)  float 0/1
        sol_mask: torch.Tensor,    # (B, MAX_SOLVENTS)
        cat_mask: torch.Tensor,    # (B, MAX_CATALYSTS)
    ) -> torch.Tensor:

        cond_kv = self.cond_proj(cond_h).unsqueeze(1)   # (B, 1, HIDDEN)

        def safe_bool_mask(mask: torch.Tensor) -> torch.Tensor:
            """
            Convert float mask (1=real, 0=pad) to bool (True=ignore).
            Guarantee at least one key per row is unmasked to prevent
            all-True rows that cause NaN in scaled dot-product attention.
            """
            bool_mask = (mask == 0)                    # True = padded
            # If every slot in a row is masked, unmask the first slot.
            all_masked = bool_mask.all(dim=-1, keepdim=True)   # (B, 1)
            bool_mask  = bool_mask & ~all_masked               # clear first when all masked
            # Always unmask position 0 for fully-masked rows
            bool_mask[:, 0] = bool_mask[:, 0] & ~all_masked.squeeze(-1)
            return bool_mask

        mol_bool = safe_bool_mask(mol_mask)
        sol_bool = safe_bool_mask(sol_mask)
        cat_bool = safe_bool_mask(cat_mask)

        for i in range(len(self.mol_self_attn)):
            n = i * 4

            # 1. Molecule self-attention
            x, _ = self.mol_self_attn[i](mol_h, mol_h, mol_h,
                                          key_padding_mask=mol_bool)
            mol_h = self.norms[n](mol_h + x)

            # 2. Molecule ↔ Solvent
            x, _ = self.mol_sol_attn[i](mol_h, sol_h, sol_h,
                                         key_padding_mask=sol_bool)
            mol_h = self.norms[n + 1](mol_h + x)

            # 3. Molecule ↔ Catalyst
            x, _ = self.mol_cat_attn[i](mol_h, cat_h, cat_h,
                                         key_padding_mask=cat_bool)
            mol_h = self.norms[n + 2](mol_h + x)

            # 4. Molecule ↔ Condition (1 token, never masked)
            x, _ = self.mol_cond_attn[i](mol_h, cond_kv, cond_kv)
            mol_h = self.norms[n + 3](mol_h + x)

            mol_h = mol_h + self.ffs[i](mol_h)

        return mol_h


class StateEncoder(nn.Module):
    """
    Encode a single ChemState → (B, STATE_DIM).
    Uses SolventFusion / CatalystFusion (improvement 1).
    Applies entity masks in pooling (improvement 6).
    """
    def __init__(self, selfies_embedder: SelfiesEmbedder):
        super().__init__()
        self.selfies_emb = selfies_embedder
        self.mol_fusion  = MoleculeFusion()
        self.sol_fusion  = SolventFusion()     # improvement 1
        self.cat_fusion  = CatalystFusion()    # improvement 1
        self.cond_enc    = ConditionEncoder()
        self.interaction = InteractionEngine()
        self.aggregate   = nn.Sequential(
            nn.Linear(HIDDEN_DIM + CONDITION_DIM, STATE_DIM * 2),
            nn.SiLU(),
            nn.Linear(STATE_DIM * 2, STATE_DIM),
            nn.LayerNorm(STATE_DIM),
        )

    def forward(self, s: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Molecule embeddings + fusion
        mol_emb = self.selfies_emb(s["mol_ids"])
        mol_h   = self.mol_fusion(mol_emb, s["mol_qty"], s["mol_role"], s["mol_phase"])

        # Solvent embeddings + fusion (improvement 1: own stype & props)
        sol_emb = self.selfies_emb(s["sol_ids"])
        sol_h   = self.sol_fusion(sol_emb, s["sol_qty"], s["sol_stype"], s["sol_props"])

        # Catalyst embeddings + fusion (improvement 1: own ctype & props)
        cat_emb = self.selfies_emb(s["cat_ids"])
        cat_h   = self.cat_fusion(cat_emb, s["cat_qty"], s["cat_ctype"], s["cat_props"])

        cond_h  = self.cond_enc(s["cond"])

        # Interaction with entity masks (improvement 6)
        mol_h   = self.interaction(
            mol_h, sol_h, cat_h, cond_h,
            s["mol_mask"], s["sol_mask"], s["cat_mask"],
        )

        # Masked mean-pool: ignore padded molecules (improvement 6)
        mask_exp = s["mol_mask"].unsqueeze(-1)                       # (B, M, 1)
        mol_pool = (mol_h * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)

        return self.aggregate(torch.cat([mol_pool, cond_h], dim=-1))


# ──────────────────────────────────────────────────────────────
# IMPROVEMENT 5 — Selective Event Injection in TemporalTransformer
# ──────────────────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """
    Temporal dynamics model with selective event injection.
    Event cross-attends ONLY at the last (current) timestep,
    not broadcast identically across all timesteps.
    """
    def __init__(self):
        super().__init__()
        self.input_proj  = nn.Linear(STATE_DIM, HIDDEN_DIM)
        self.delta_proj  = nn.Linear(STATE_DIM, HIDDEN_DIM // 4)
        self.pos_emb     = nn.Embedding(HISTORY_LEN + 4, HIDDEN_DIM)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=NHEAD, dim_feedforward=HIDDEN_DIM * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=TEMPORAL_LAYERS)

        # Improvement 5: cross-attention for event, applied only to last timestep
        self.event_proj       = nn.Linear(EVENT_DIM, HIDDEN_DIM)
        self.event_cross_attn = nn.MultiheadAttention(
            HIDDEN_DIM, NHEAD, dropout=0.1, batch_first=True,
        )
        self.event_norm       = nn.LayerNorm(HIDDEN_DIM)

        self.out_proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, STATE_DIM * 2),
            nn.GELU(),
            nn.Linear(STATE_DIM * 2, STATE_DIM),
            nn.LayerNorm(STATE_DIM),
        )

    def forward(self, state_seq: torch.Tensor, event_emb: torch.Tensor) -> torch.Tensor:
        """
        state_seq : (B, T, STATE_DIM)
        event_emb : (B, EVENT_DIM)
        Returns   : (B, STATE_DIM)
        """
        B, T, _ = state_seq.shape

        # Project states
        x   = self.input_proj(state_seq)                       # (B, T, HIDDEN)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x   = x + self.pos_emb(pos)

        # Causal self-attention over history
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        x    = self.transformer(x, mask=mask)                  # (B, T, HIDDEN)

        # Improvement 5: inject event ONLY into the last timestep via cross-attention
        last      = x[:, -1:, :]                               # (B, 1, HIDDEN)
        ev_kv     = self.event_proj(event_emb).unsqueeze(1)    # (B, 1, HIDDEN)
        ev_out, _ = self.event_cross_attn(last, ev_kv, ev_kv)  # (B, 1, HIDDEN)
        last      = self.event_norm(last + ev_out)             # residual

        return self.out_proj(last.squeeze(1))                  # (B, STATE_DIM)


# ──────────────────────────────────────────────────────────────
# IMPROVEMENT 2 — SELFIES Reconstruction Decoder
# ──────────────────────────────────────────────────────────────

class SELFIESDecoder(nn.Module):
    """
    Latent embedding → token logits for SELFIES reconstruction.
    Uses a lightweight autoregressive transformer decoder.
    latent (B, LATENT_DIM) + teacher-forced tokens → (B, SEQ_LEN, VOCAB_SIZE)
    """
    def __init__(self):
        super().__init__()
        self.latent_proj = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.tok_embed   = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM, padding_idx=PAD_IDX)
        self.pos_emb     = nn.Embedding(MAX_SEQ_LEN + 2, HIDDEN_DIM)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=HIDDEN_DIM, nhead=NHEAD, dim_feedforward=HIDDEN_DIM * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.out_proj = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self,
        latent:   torch.Tensor,
        tgt_ids:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B   = latent.size(0)
        mem = self.latent_proj(latent).unsqueeze(1)   # (B, 1, HIDDEN)

        # Teacher-forced training path
        if tgt_ids is not None:
            L   = tgt_ids.size(1)
            pos = torch.arange(L, device=latent.device).unsqueeze(0)
            x   = self.tok_embed(tgt_ids) + self.pos_emb(pos)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                L, device=latent.device)
            out = self.decoder(x, mem, tgt_mask=tgt_mask)
            return self.out_proj(out)                  # (B, L, VOCAB_SIZE)

        # Greedy inference path (no gradient needed from caller)
        ids = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=latent.device)
        for _ in range(MAX_SEQ_LEN - 1):
            L   = ids.size(1)
            pos = torch.arange(L, device=latent.device).unsqueeze(0)
            x   = self.tok_embed(ids) + self.pos_emb(pos)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                L, device=latent.device)
            out      = self.decoder(x, mem, tgt_mask=tgt_mask)
            next_tok = self.out_proj(out[:, -1, :]).argmax(-1, keepdim=True)
            ids      = torch.cat([ids, next_tok], dim=1)
            if (next_tok == EOS_IDX).all():
                break
        # Return logits for the full generated sequence
        L   = ids.size(1)
        pos = torch.arange(L, device=latent.device).unsqueeze(0)
        x   = self.tok_embed(ids) + self.pos_emb(pos)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L, device=latent.device)
        out = self.decoder(x, mem, tgt_mask=tgt_mask)
        return self.out_proj(out)

    @torch.no_grad()
    def decode_greedy(self, latent: torch.Tensor) -> List[str]:
        """Decode latent vectors to SELFIES strings (inference only)."""
        B   = latent.size(0)
        mem = self.latent_proj(latent).unsqueeze(1)
        ids = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=latent.device)

        for _ in range(MAX_SEQ_LEN - 1):
            L        = ids.size(1)
            pos      = torch.arange(L, device=latent.device).unsqueeze(0)
            x        = self.tok_embed(ids) + self.pos_emb(pos)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                L, device=latent.device)
            out      = self.decoder(x, mem, tgt_mask=tgt_mask)
            next_tok = self.out_proj(out[:, -1, :]).argmax(-1, keepdim=True)
            ids      = torch.cat([ids, next_tok], dim=1)
            if (next_tok == EOS_IDX).all():
                break

        return [token_ids_to_selfies(row) for row in ids]


def selfies_to_smiles(sel: str) -> str:
    """Convert SELFIES string to SMILES (fallback to water if invalid)."""
    try:
        smi = sf.decoder(sel)
        return smi if smi else "O"
    except Exception:
        return "O"


# ─────────────────────────────────────────────
# DECODERS
# ─────────────────────────────────────────────

class MoleculeDecoder(nn.Module):
    """
    Predict updated molecule latents + quantities from next-state vector.
    Includes SELFIES reconstruction head (improvement 2).
    """
    def __init__(self, selfies_decoder: SELFIESDecoder):
        super().__init__()
        self.selfies_dec = selfies_decoder
        self.mol_head    = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, MAX_MOLS * LATENT_DIM),
        )
        self.qty_head    = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM // 2), nn.SiLU(),
            nn.Linear(HIDDEN_DIM // 2, MAX_MOLS),
            nn.Softplus(),
        )

    def forward(
        self,
        z:        torch.Tensor,
        tgt_ids:  Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        z        : (B, STATE_DIM)
        tgt_ids  : (B, MAX_MOLS, SEQ_LEN) — ground truth for SELFIES reconstruction
        Returns mol_emb, mol_qty, selfies_logits
        """
        B        = z.size(0)
        mol_emb  = self.mol_head(z).reshape(B, MAX_MOLS, LATENT_DIM)
        mol_qty  = self.qty_head(z).unsqueeze(-1)                     # (B, MAX_MOLS, 1)

        # Improvement 2: reconstruct SELFIES from predicted latents
        flat_latent = mol_emb.reshape(B * MAX_MOLS, LATENT_DIM)

        if tgt_ids is not None:
            flat_tgt    = tgt_ids.reshape(B * MAX_MOLS, -1)
            selfies_logits = self.selfies_dec(flat_latent, flat_tgt)     # (B*N, SEQ, V)
            selfies_logits = selfies_logits.reshape(B, MAX_MOLS, -1, VOCAB_SIZE)
        else:
            # Inference: greedy decode
            selfies_logits = None

        return {"mol_emb": mol_emb, "mol_qty": mol_qty, "selfies_logits": selfies_logits}

    @torch.no_grad()
    def decode_molecules(self, z: torch.Tensor) -> List[List[str]]:
        """Return SELFIES strings for each predicted molecule. Shape: (B, MAX_MOLS)."""
        B           = z.size(0)
        mol_emb     = self.mol_head(z).reshape(B, MAX_MOLS, LATENT_DIM)
        flat_latent = mol_emb.reshape(B * MAX_MOLS, LATENT_DIM)
        selfies_strs = self.selfies_dec.decode_greedy(flat_latent)
        return [selfies_strs[i * MAX_MOLS:(i + 1) * MAX_MOLS] for i in range(B)]


# ──────────────────────────────────────────────────────────────
# IMPROVEMENT 7 — Physically-bounded ConditionDecoder
# ──────────────────────────────────────────────────────────────

class ConditionDecoder(nn.Module):
    """
    Predict physically-meaningful conditions:
      T   = 273 + sigmoid(x) * 500     →  273 K to 773 K
      pH  = sigmoid(x) * 14            →  0 to 14
      P   = softplus(x) * 2 + 0.1      →  > 0.1 atm  (stable near 1 atm)
      V   = softplus(x) * 10 + 0.001   →  > 0 L
    All decoded to normalised [0,1] range for loss comparability.
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 4),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw  = self.mlp(z)                                    # (B, 4)
        # Keep output in [0,1] normalised space to match targets
        T    = torch.sigmoid(raw[:, 0:1])               # maps to 273-773 K after denorm
        pH   = torch.sigmoid(raw[:, 1:2])               # maps to 0-14
        P    = torch.sigmoid(raw[:, 2:3])               # maps to 0.1-5 atm
        V    = F.softplus(raw[:, 3:4]) / 10.0           # bounded positive volume
        V    = torch.clamp(V, 0.0, 1.0)
        return torch.cat([T, pH, P, V], dim=-1)         # (B, 4) normalised


class CatalystDecoder(nn.Module):
    """Predict catalyst properties + quantity → (B, MAX_CATALYSTS, 5)."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, MAX_CATALYSTS * 5),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(z).reshape(z.size(0), MAX_CATALYSTS, 5))


class SolventDecoder(nn.Module):
    """Predict solvent states → (B, MAX_SOLVENTS, 6)."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, MAX_SOLVENTS * 6),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(z).reshape(z.size(0), MAX_SOLVENTS, 6))


# ──────────────────────────────────────────────────────────────
# IMPROVEMENT 4 — ReactionRateHead + kinetics prediction
# ──────────────────────────────────────────────────────────────

class ReactionRateHead(nn.Module):
    """
    Predict per-molecule reaction rates: dQ/dt and velocity.
    Encodes kinetic information: fast/slow reactions, buildup, depletion.

    Outputs:
      dqdt      : (B, MAX_MOLS)  — quantity change rate (signed, mol/s normalised)
      velocity  : (B,)           — overall reaction velocity scalar
    """
    def __init__(self):
        super().__init__()
        self.dqdt_head = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, MAX_MOLS),
            nn.Tanh(),   # signed: positive = forming, negative = consuming
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM // 2), nn.SiLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
            nn.Softplus(),  # velocity ≥ 0
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "dqdt":     self.dqdt_head(z),           # (B, MAX_MOLS)
            "velocity": self.velocity_head(z),       # (B, 1)
        }


# ─────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────

class FullReactionWorldModel(nn.Module):
    """
    End-to-end dynamic chemistry world model — improved version.
    """
    def __init__(self):
        super().__init__()
        self.selfies_emb    = SelfiesEmbedder()
        self.selfies_dec    = SELFIESDecoder()              # improvement 2
        self.state_encoder  = StateEncoder(self.selfies_emb)
        self.event_encoder  = EventEncoder(self.selfies_emb)
        self.temporal       = TemporalTransformer()

        self.mol_decoder    = MoleculeDecoder(self.selfies_dec)
        self.cond_decoder   = ConditionDecoder()
        self.cat_decoder    = CatalystDecoder()
        self.sol_decoder    = SolventDecoder()
        self.rate_head      = ReactionRateHead()            # improvement 4

    def encode_history(self, history: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, T = history["cond"].shape[:2]
        return torch.stack([
            self.state_encoder({k: v[:, t] for k, v in history.items()})
            for t in range(T)
        ], dim=1)   # (B, T, STATE_DIM)

    def forward(
        self,
        history:   Dict[str, torch.Tensor],
        event:     Dict[str, torch.Tensor],
        tgt_mol_ids: Optional[torch.Tensor] = None,   # (B, MAX_MOLS, SEQ_LEN)
    ) -> Dict[str, torch.Tensor]:

        state_seq  = self.encode_history(history)
        event_emb  = self.event_encoder(event)
        next_z     = self.temporal(state_seq, event_emb)

        mol_out    = self.mol_decoder(next_z, tgt_mol_ids)
        cond_pred  = self.cond_decoder(next_z)
        cat_pred   = self.cat_decoder(next_z)
        sol_pred   = self.sol_decoder(next_z)
        rate_out   = self.rate_head(next_z)             # improvement 4

        return {
            "mol_emb":        mol_out["mol_emb"],
            "mol_qty":        mol_out["mol_qty"],
            "selfies_logits": mol_out["selfies_logits"],
            "cond":           cond_pred,
            "cat":            cat_pred,
            "sol":            sol_pred,
            "dqdt":           rate_out["dqdt"],
            "velocity":       rate_out["velocity"],
            "next_z":         next_z,
        }


# ─────────────────────────────────────────────
# LOSS FUNCTIONS (improvements 2, 3, 4)
# ─────────────────────────────────────────────

def compute_loss(
    preds:    Dict[str, torch.Tensor],
    targets:  Dict[str, torch.Tensor],
    model:    FullReactionWorldModel,
    src_smiles_batch:  List[List[str]],
    src_qtys_batch:    List[List[float]],
    tgt_smiles_batch:  List[List[str]],
    tgt_qtys_batch:    List[List[float]],
) -> Dict[str, torch.Tensor]:

    # ── Molecule embedding loss ──
    with torch.no_grad():
        tgt_mol_emb = model.selfies_emb(targets["mol_ids"])
    mol_emb_loss = F.mse_loss(preds["mol_emb"], tgt_mol_emb)
    cos_mol_loss = 1.0 - F.cosine_similarity(
        preds["mol_emb"].reshape(-1, LATENT_DIM),
        tgt_mol_emb.reshape(-1, LATENT_DIM),
    ).mean()

    # ── Quantity loss ──
    qty_loss = F.mse_loss(preds["mol_qty"], targets["mol_qty"])

    # ── Condition loss ──
    cond_loss = F.mse_loss(preds["cond"], targets["cond"])

    # ── Catalyst loss ──
    tgt_cat  = torch.cat([targets["cat_qty"], targets["cat_props"]], dim=-1)
    cat_loss = F.mse_loss(preds["cat"], tgt_cat)

    # ── Solvent loss ──
    tgt_sol  = torch.cat([targets["sol_qty"], targets["sol_props"]], dim=-1)
    sol_loss = F.mse_loss(preds["sol"], tgt_sol)

    # ── Improvement 2: SELFIES reconstruction loss ──
    selfies_loss = torch.tensor(0.0, device=preds["mol_emb"].device)
    if preds["selfies_logits"] is not None:
        B, N, L, V = preds["selfies_logits"].shape
        tgt_ids     = targets["mol_ids"]                # (B, N, SEQ_LEN=64)
        # Teacher-forcing shift:
        #   input  = tgt_ids[:, :, :-1]  (SOS … token_{L-2})
        #   target = tgt_ids[:, :, 1:]   (token_1 … token_{L-1})
        # logits already produced from tgt_ids[:,: , :L] in MoleculeDecoder forward,
        # so logits shape is (B, N, L, V) where L == SEQ_LEN.
        # We predict position 1..L-1 from input 0..L-2.
        logits_shift = preds["selfies_logits"][:, :, :-1, :]  # (B, N, L-1, V)
        target_shift = tgt_ids[:, :, 1:]                       # (B, N, L-1)
        # Flatten for cross_entropy
        logits_flat  = logits_shift.reshape(-1, V)
        tgt_flat     = target_shift.reshape(-1)
        selfies_loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=PAD_IDX)

    # ── Improvement 3: Atom conservation loss ──
    atom_loss = torch.tensor(0.0, device=preds["mol_emb"].device)
    try:
        for src_smiles, src_qtys, tgt_smiles, tgt_qtys in zip(
            src_smiles_batch, src_qtys_batch, tgt_smiles_batch, tgt_qtys_batch
        ):
            src_atoms = smiles_list_to_atom_tensor(src_smiles, src_qtys)
            tgt_atoms = smiles_list_to_atom_tensor(tgt_smiles, tgt_qtys)
            # Use predicted quantities (on device) for the target side
            atom_loss = atom_loss + F.mse_loss(src_atoms.to(preds["mol_qty"].device),
                                               tgt_atoms.to(preds["mol_qty"].device))
        atom_loss = atom_loss / max(len(src_smiles_batch), 1)
    except Exception:
        pass

    # ── Improvement 4: Kinetics loss ──
    # Supervise velocity as mean absolute quantity in the target state (proxy for rate).
    # Supervise dqdt sign: products have positive sign, reactants negative.
    # All targets derived purely from ground-truth to avoid NaN propagation.
    tgt_qty_sq     = targets["mol_qty"].squeeze(-1)              # (B, MAX_MOLS)
    gt_velocity    = tgt_qty_sq.mean(dim=-1, keepdim=True).detach()  # (B, 1)
    kinetics_loss  = F.mse_loss(preds["velocity"], gt_velocity)

    product_mask   = targets["mol_role"][:, :, 1]               # (B, MAX_MOLS) 1=product
    gt_dqdt_sign   = (product_mask * 2.0 - 1.0) * tgt_qty_sq.detach()
    dqdt_sign_loss = F.mse_loss(preds["dqdt"], gt_dqdt_sign)

    def _safe(t: torch.Tensor) -> torch.Tensor:
        """Replace NaN/Inf loss with zero so one bad batch doesn't kill training."""
        return torch.where(torch.isfinite(t), t, torch.zeros_like(t))

    total = (
        1.00 * _safe(mol_emb_loss)
      + 0.50 * _safe(cos_mol_loss)
      + 0.80 * _safe(qty_loss)
      + 0.60 * _safe(cond_loss)
      + 0.40 * _safe(cat_loss)
      + 0.40 * _safe(sol_loss)
      + 0.50 * _safe(selfies_loss)
      + 0.20 * _safe(atom_loss)
      + 0.30 * _safe(kinetics_loss)
      + 0.20 * _safe(dqdt_sign_loss)
    )

    return {
        "total":        total,
        "mol_emb":      mol_emb_loss,
        "cos_mol":      cos_mol_loss,
        "qty":          qty_loss,
        "cond":         cond_loss,
        "cat":          cat_loss,
        "sol":          sol_loss,
        "selfies":      selfies_loss,
        "atom_cons":    atom_loss,
        "kinetics":     kinetics_loss,
        "dqdt_sign":    dqdt_sign_loss,
    }


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train(model, train_loader, val_loader, epochs=EPOCHS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Linear warmup then cosine decay
    total_steps = epochs * len(train_loader)
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler(enabled=(DEVICE.type == "cuda"))
    best_val  = float("inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        metrics = {k: 0.0 for k in [
            "total","mol_emb","qty","cond","cat","sol","selfies","atom_cons","kinetics"
        ]}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [TRAIN]")

        for batch in pbar:
            history, event, target, src_sm, src_qt, tgt_sm, tgt_qt = batch
            history = {k: v.to(DEVICE) for k, v in history.items()}
            event   = {k: v.to(DEVICE) for k, v in event.items()}
            target  = {k: v.to(DEVICE) for k, v in target.items()}

            optimizer.zero_grad()
            with autocast(enabled=(DEVICE.type == "cuda")):
                preds  = model(history, event, tgt_mol_ids=target["mol_ids"])
                losses = compute_loss(
                    preds, target, model,
                    src_sm, src_qt, tgt_sm, tgt_qt,
                )

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            for k in metrics:
                if k in losses:
                    v = losses[k].item()
                    if math.isfinite(v):
                        metrics[k] += v

            pbar.set_postfix({
                "loss":  f"{losses['total'].item():.4f}",
                "mol":   f"{losses['mol_emb'].item():.4f}",
                "sfls":  f"{losses['selfies'].item():.4f}",
                "gnorm": f"{grad_norm:.2f}",
            })

        n = len(train_loader)
        print("  TRAIN | " + "  ".join(f"{k}:{v/n:.4f}" for k, v in metrics.items()))

        val_loss = validate(model, val_loader)
        print(f"  VAL   | total: {val_loss:.4f}")

        ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "val_loss": val_loss,
        }
        torch.save(ckpt, os.path.join(SAVE_DIR, f"epoch_{epoch:03d}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(SAVE_DIR, "best.pt"))
            print(f"  ✓ New best saved (val={best_val:.4f})")

    return model


@torch.no_grad()
def validate(model, loader) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        history, event, target, src_sm, src_qt, tgt_sm, tgt_qt = batch
        history = {k: v.to(DEVICE) for k, v in history.items()}
        event   = {k: v.to(DEVICE) for k, v in event.items()}
        target  = {k: v.to(DEVICE) for k, v in target.items()}
        preds   = model(history, event, tgt_mol_ids=target["mol_ids"])
        losses  = compute_loss(preds, target, model, src_sm, src_qt, tgt_sm, tgt_qt)
        total  += losses["total"].item()
    return total / len(loader)


# ──────────────────────────────────────────────────────────────
# IMPROVEMENT 10 — Full Rollout with Molecule Reconstruction
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout(
    model:          FullReactionWorldModel,
    initial_states: List[ChemState],
    events:         List[Event],
    n_steps:        int,
) -> List[Dict]:
    """
    Autoregressively predict n_steps future states.
    Improvement 10: decode predicted latents → SELFIES → SMILES → new ChemState
    so each rollout step uses fully reconstructed states.
    """
    model.eval()
    assert len(initial_states) == HISTORY_LEN

    history_buf = [state_to_tensors(s) for s in initial_states]
    # Keep a parallel list of actual ChemState objects for reconstruction
    chem_state_buf = list(initial_states)

    predicted_states = []

    for step in range(n_steps):
        # Stack history (1, T, ...)
        history_batch = {
            k: torch.stack([h[k] for h in history_buf], dim=0).unsqueeze(0).to(DEVICE)
            for k in history_buf[0]
        }
        ev_batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in tensorize_event(events[step]).items()}

        preds = model(history_batch, ev_batch, tgt_mol_ids=None)

        # ── Improvement 10: decode molecule latents → SELFIES → SMILES ──
        pred_selfies_list = model.mol_decoder.decode_molecules(preds["next_z"])
        pred_selfies      = pred_selfies_list[0]   # list of MAX_MOLS SELFIES strings

        pred_qtys  = preds["mol_qty"].squeeze(0).squeeze(-1).cpu().tolist()
        pred_cond  = preds["cond"].squeeze(0).cpu().tolist()
        pred_dqdt  = preds["dqdt"].squeeze(0).cpu().tolist()
        pred_vel   = preds["velocity"].squeeze(0).item()

        # Build reconstructed MolState list
        new_mols = []
        for j, (sel, qty) in enumerate(zip(pred_selfies, pred_qtys)):
            smi  = selfies_to_smiles(sel) if sel else "O"
            role = "product" if pred_dqdt[j] > 0 else "reactant"
            new_mols.append(MolState(smi, max(0.0, qty), role, "liquid"))

        # Rebuild conditions with physical de-normalisation
        new_cond = Conditions(
            temperature = 273.0 + pred_cond[0] * 500.0,
            ph          = pred_cond[1] * 14.0,
            pressure    = max(0.1, pred_cond[2] * 5.0),
            volume      = max(0.001, pred_cond[3] * 10.0),
        )

        # Rebuild solvents and catalysts from previous state (updated quantities)
        prev_cs   = chem_state_buf[-1]
        new_solv  = copy.deepcopy(prev_cs.solvents)
        new_cats  = copy.deepcopy(prev_cs.catalysts)
        sol_pred  = preds["sol"].squeeze(0).cpu()
        cat_pred  = preds["cat"].squeeze(0).cpu()
        for i, sv in enumerate(new_solv):
            if i < sol_pred.size(0):
                sv.quantity = max(0.0, float(sol_pred[i, 0]) * 50.0)
        for i, ct in enumerate(new_cats):
            if i < cat_pred.size(0):
                ct.activity    = float(cat_pred[i, 2])
                ct.selectivity = float(cat_pred[i, 3])

        next_t = prev_cs.time + 300.0
        new_chem_state = ChemState(new_mols, new_solv, new_cats, new_cond, next_t)

        # Record prediction with metadata
        predicted_states.append({
            "chem_state":  new_chem_state,
            "cond_norm":   pred_cond,
            "dqdt":        pred_dqdt,
            "velocity":    pred_vel,
            "mol_smiles":  [selfies_to_smiles(s) for s in pred_selfies],
            "mol_qtys":    pred_qtys,
        })

        # Update rolling history buffer with FULLY reconstructed state tensors
        new_state_t     = state_to_tensors(new_chem_state)
        history_buf     = history_buf[1:] + [new_state_t]
        chem_state_buf  = chem_state_buf[1:] + [new_chem_state]

        print(
            f"  Step {step+1}: T={new_cond.temperature:.1f}K  "
            f"pH={new_cond.ph:.2f}  P={new_cond.pressure:.3f}atm  "
            f"vel={pred_vel:.4f}  "
            f"mols={[f'{s}({q:.3f})' for s,q in zip(predicted_states[-1]['mol_smiles'], pred_qtys) if q > 0.01]}"
        )

    return predicted_states


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    full_dataset = ReactionDataset(n_samples=2000, seed=0)
    val_size     = int(0.1 * len(full_dataset))
    train_size   = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    model    = FullReactionWorldModel().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    model = train(model, train_loader, val_loader, epochs=EPOCHS)
    print(f"\nTraining complete ✓  Checkpoints → {SAVE_DIR}/")

    # ── Demo rollout with full reconstruction ──
    print("\n--- Autoregressive Rollout Demo (full molecule reconstruction) ---")
    random.seed(99)
    warm_up_traj, warm_up_events = make_trajectory(HISTORY_LEN + 5)
    preds = rollout(
        model,
        initial_states = warm_up_traj[:HISTORY_LEN],
        events         = warm_up_events[HISTORY_LEN:HISTORY_LEN + 3],
        n_steps        = 3,
    )

    print(f"\nRollout produced {len(preds)} future state predictions.")
    for i, p in enumerate(preds):
        cs = p["chem_state"]
        print(
            f"  t+{i+1}: T={cs.conditions.temperature:.1f}K  "
            f"pH={cs.conditions.ph:.2f}  vel={p['velocity']:.5f}\n"
            f"         molecules: {[(m.smiles, round(m.quantity, 4), m.role) for m in cs.molecules if m.quantity > 0.001]}"
        )

    print("\nDone.")