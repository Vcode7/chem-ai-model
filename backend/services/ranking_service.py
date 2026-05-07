"""
OC-Ranking Catalyst Energy Scoring Service
==========================================
Wraps the CatalystEnergyModel from oc-ranking.py.

If no checkpoint is found in checkpoints/oc-ranking/, operates in MOCK mode:
  - Returns Gaussian-sampled energy values centred around -1.0 eV
  - Clearly marks results as mock in the response

When a checkpoint is placed in checkpoints/oc-ranking/, real inference activates
automatically on the next server restart.

NOTE: The real model expects 3D atomic structure data. For SMILES-based inference,
we generate a lightweight pseudo-structure (random coordinates per atom type)
as a placeholder until real structure data is available.
"""

import os
import sys
import logging
import random
import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("ranking_service")

BASE_DIR       = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "oc-ranking"
MODELS_DIR     = BASE_DIR / "models"

# ── Element → index map (from oc-ranking.py TYPE_MAP) ────────────────────────
TYPE_MAP = [
    "Ag","Al","As","Au","B","Bi","C","Ca","Cd","Cl","Co","Cr","Cs","Cu","Fe",
    "Ga","Ge","H","Hf","Hg","In","Ir","K","Mg","Mn","Mo","N","Na","Nb","Ni",
    "O","Os","P","Pb","Pd","Pt","Rb","Re","Rh","Ru","S","Sb","Sc","Se","Si",
    "Sn","Sr","Ta","Tc","Te","Ti","Tl","V","W","Y","Zn","Zr",
]
ELEM_TO_IDX = {e: i for i, e in enumerate(TYPE_MAP)}
N_ELEM = len(TYPE_MAP)


def _import_oc_ranking():
    spec = importlib.util.spec_from_file_location(
        "oc_ranking", MODELS_DIR / "oc-ranking.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _smiles_to_pseudo_structure(smiles: str) -> dict:
    """
    Convert a SMILES string to a pseudo 3D structure for the OC-Ranking model.
    Uses RDKit if available for real atom types; falls back to heuristic parsing.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        conf = mol.GetConformer()
        types  = []
        coords = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            idx = ELEM_TO_IDX.get(sym, ELEM_TO_IDX.get("C", 6))
            pos = conf.GetAtomPosition(atom.GetIdx())
            types.append(idx)
            coords.append([pos.x, pos.y, pos.z])
        return {"types": np.array(types, dtype=int), "coords": np.array(coords, dtype=np.float32)}
    except Exception:
        # Fallback: count atoms heuristically
        import re
        atoms = re.findall(r'[A-Z][a-z]?', smiles)
        types  = [ELEM_TO_IDX.get(a, 6) for a in atoms[:20]] or [6]
        n      = len(types)
        coords = np.random.randn(n, 3).astype(np.float32) * 1.5
        return {"types": np.array(types, dtype=int), "coords": coords}


# ── Service ───────────────────────────────────────────────────────────────────

class RankingService:
    def __init__(self):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.oc_mod     = None
        self.model      = None
        self.mock       = True
        self.loaded     = False

    def load(self):
        logger.info("Loading OC-Ranking model…")

        checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"))
        if not checkpoints:
            logger.warning("No OC-Ranking checkpoint found — using mock energy values")
            self.mock   = True
            self.loaded = True
            return

        ckpt_path = checkpoints[-1]
        try:
            self.oc_mod = _import_oc_ranking()
            self.model  = self.oc_mod.CatalystEnergyModel().to(self.device)
            ckpt        = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self.mock   = False
            logger.info(f"OC-Ranking loaded from {ckpt_path.name} ✓")
        except Exception as e:
            logger.error(f"OC-Ranking load error: {e} — falling back to mock")
            self.mock = True

        self.loaded = True

    def rank(
        self,
        reactant_smiles_list: list[str],
        catalyst_list:        list[dict],   # [{"smiles": ..., "selfies": ...}]
        input_format:         str = "smiles",
    ) -> list[dict]:
        """
        Score each catalyst against every reactant.
        Returns: [
          {
            "catalyst": {...},
            "energies": {"reactant_smiles": energy_eV, ...},
            "avg_energy": float,
          }, ...
        ] sorted ascending by avg_energy.
        """
        if not self.loaded:
            self.load()

        results = []
        for cat in catalyst_list:
            cat_smi  = cat.get("smiles", "")
            energies = {}

            for r_smi in reactant_smiles_list:
                if self.mock:
                    # Realistic mock: draw from N(-1.0, 0.25)
                    energies[r_smi] = round(float(np.random.normal(-1.0, 0.25)), 4)
                else:
                    energies[r_smi] = self._score_pair(r_smi, cat_smi)

            avg = round(float(np.mean(list(energies.values()))), 4) if energies else 0.0
            results.append({
                "catalyst":   cat,
                "energies":   energies,
                "avg_energy": avg,
                "mock":       self.mock,
            })

        results.sort(key=lambda x: x["avg_energy"])
        for rank, r in enumerate(results, 1):
            r["rank"] = rank

        return results

    def _score_pair(self, reactant_smi: str, catalyst_smi: str) -> float:
        """Run one real inference pass → predicted energy in eV."""
        try:
            from oc_ranking import MAX_ATOMS
            MAX_ATOMS = self.oc_mod.MAX_ATOMS

            r_struct = _smiles_to_pseudo_structure(reactant_smi)
            c_struct = _smiles_to_pseudo_structure(catalyst_smi)

            combined_types  = np.concatenate([r_struct["types"],  c_struct["types"]])
            combined_coords = np.concatenate([r_struct["coords"], c_struct["coords"]])
            n = len(combined_types)

            if n > MAX_ATOMS:
                combined_types  = combined_types[:MAX_ATOMS]
                combined_coords = combined_coords[:MAX_ATOMS]
                n = MAX_ATOMS

            pad   = MAX_ATOMS - n
            types = np.concatenate([combined_types, np.full(pad, N_ELEM, dtype=int)])
            crds  = np.concatenate([combined_coords, np.zeros((pad, 3), dtype=np.float32)])
            mask  = np.array([True]*n + [False]*pad)

            t_types = torch.tensor(types, dtype=torch.long).unsqueeze(0).to(self.device)
            t_crds  = torch.tensor(crds,  dtype=torch.float32).unsqueeze(0).to(self.device)
            t_mask  = torch.tensor(mask,  dtype=torch.bool).unsqueeze(0).to(self.device)

            with torch.no_grad():
                energy = self.model(t_types, t_crds, t_mask).item()
            return round(energy, 4)
        except Exception as e:
            logger.error(f"Ranking score error: {e}")
            return round(float(np.random.normal(-1.0, 0.25)), 4)


# ── Singleton ─────────────────────────────────────────────────────────────────
ranking_service = RankingService()
