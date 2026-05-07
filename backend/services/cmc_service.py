"""
CMC Catalyst Prediction Service
================================
Wraps the three pretrained CatalystModel variants (A, B, C) from cmc.py.

Auto-selects model:
  - solvents only  → Model A
  - products only  → Model B
  - both           → Model C
  - neither        → Model A (fallback)

Handles both SMILES and SELFIES input strings.
"""

import os
import sys
import logging
import random
from pathlib import Path
from typing import Optional
import importlib.util

import torch
import selfies as sf

from services.vocab_service import vocab_service

logger = logging.getLogger("cmc_service")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "cmc"
MODELS_DIR = BASE_DIR / "models"

# ── Dynamically import cmc.py (not a package) ────────────────────────────────
def _import_cmc():
    spec = importlib.util.spec_from_file_location("cmc", MODELS_DIR / "cmc.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── Helpers ───────────────────────────────────────────────────────────────────

def selfies_to_smiles(s: str) -> Optional[str]:
    """Convert SELFIES string to SMILES. Returns None on failure."""
    try:
        smi = sf.decoder(s)
        return smi if smi else None
    except Exception:
        return None

def smiles_to_selfies(s: str) -> Optional[str]:
    """Convert SMILES string to SELFIES. Returns None on failure."""
    try:
        return sf.encoder(s)
    except Exception:
        return None

def normalise_to_smiles(molecule: str, input_format: str) -> Optional[str]:
    """
    Normalise a molecule string (SMILES or SELFIES) to SMILES.
    input_format: 'smiles' | 'selfies'
    """
    if input_format == "selfies":
        return selfies_to_smiles(molecule)
    return molecule  # already SMILES


# ── Main service class ────────────────────────────────────────────────────────

class CMCService:
    """
    Loads CMC models A, B, C from checkpoints and exposes a predict() method.
    Falls back to a mock if checkpoints are missing.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cmc    = None   # module reference
        self.models: dict[str, object] = {}   # "A", "B", "C"
        self.loaded = False
        self.mock   = False

    def load(self):
        """Called at app startup. Loads all three models."""
        logger.info("Loading CMC models…")
        try:
            self.cmc = _import_cmc()
        except Exception as e:
            logger.error(f"Failed to import cmc.py: {e}")
            self.mock = True
            self.loaded = True
            return

        # Use vocab from checkpoint if available; fall back to vocab_service
        fallback_stoi = vocab_service.stoi
        fallback_itos = vocab_service.itos
        fallback_size = vocab_service.vocab_size

        for variant in ("A", "B", "C"):
            # Find latest phase-2 checkpoint for this variant
            checkpoints = sorted(CHECKPOINT_DIR.glob(f"catalyst_{variant}_phase2_*.pt"))
            if not checkpoints:
                checkpoints = sorted(CHECKPOINT_DIR.glob(f"catalyst_{variant}_*.pt"))
            if not checkpoints:
                logger.warning(f"No checkpoint found for Model {variant} — will use mock")
                continue

            ckpt_path = checkpoints[-1]
            logger.info(f"Loading Model {variant} from {ckpt_path.name}…")
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                saved_stoi = ckpt.get("vocab", {})

                if saved_stoi:
                    # Checkpoint has its own vocabulary
                    saved_itos = {i: t for t, i in saved_stoi.items()}
                    vocab_size = len(saved_stoi)
                    logger.info(f"  Model {variant}: using checkpoint vocab ({vocab_size} tokens)")
                elif fallback_stoi:
                    # Use training-data vocabulary from vocab_service
                    saved_stoi = fallback_stoi
                    saved_itos = fallback_itos
                    vocab_size = fallback_size
                    logger.info(
                        f"  Model {variant}: checkpoint has no vocab — "
                        f"using training vocab ({vocab_size} tokens)"
                    )
                else:
                    logger.warning(f"Checkpoint {variant} has no vocab and vocab_service empty — skipping")
                    continue

                gin         = self.cmc.GINEncoder().to(self.device)
                transformer = self.cmc.CatalystTransformer().to(self.device)
                decoder     = self.cmc.SelfiesDecoder(vocab_size, saved_stoi, saved_itos).to(self.device)

                gin.load_state_dict(ckpt["gin_state_dict"])
                transformer.load_state_dict(ckpt["transformer_state_dict"])
                decoder.load_state_dict(ckpt["decoder_state_dict"])

                model = self.cmc.CatalystModel(gin, transformer, decoder).to(self.device)
                model.eval()
                self.models[variant] = model
                logger.info(f"Model {variant} loaded ✓ (epoch {ckpt.get('epoch', '?')})")
            except Exception as e:
                logger.error(f"Error loading Model {variant}: {e}")

        if not self.models:
            logger.warning("No CMC models loaded — using mock mode")
            self.mock = True

        self.loaded = True
        logger.info("CMC service ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        reactants:    list[str],
        solvents:     list[str],
        products:     list[str],
        input_format: str = "smiles",   # 'smiles' | 'selfies'
        conf_threshold: float = 0.0,
        n_candidates: int = 8,
    ) -> dict:
        """
        Predict candidate catalysts.

        Returns:
          {
            "model_used": "A" | "B" | "C",
            "catalysts":  [{"smiles": ..., "selfies": ...}, ...]
          }
        """
        if not self.loaded:
            self.load()

        # Convert all inputs to SMILES
        r_smiles = [s for m in reactants if (s := normalise_to_smiles(m, input_format))]
        s_smiles = [s for m in solvents  if (s := normalise_to_smiles(m, input_format))]
        p_smiles = [s for m in products  if (s := normalise_to_smiles(m, input_format))]

        if not r_smiles:
            return {"model_used": None, "catalysts": [], "error": "No valid reactants"}

        # Choose model variant
        has_solvents = bool(s_smiles)
        has_products = bool(p_smiles)

        if has_solvents and has_products:
            variant = "C"
        elif has_products:
            variant = "B"
        else:
            variant = "A"   # solvents-only or neither → A

        # Mock mode
        if self.mock or variant not in self.models:
            return self._mock_predict(variant, n_candidates)

        # Real prediction
        try:
            input_smiles = r_smiles + s_smiles + p_smiles
            model = self.models[variant]
            with torch.no_grad():
                cat_smiles_list = model.predict(input_smiles, conf_threshold=conf_threshold)

            if not cat_smiles_list:
                return self._mock_predict(variant, n_candidates)

            results = []
            for smi in cat_smiles_list[:n_candidates]:
                sel = smiles_to_selfies(smi) or ""
                results.append({"smiles": smi, "selfies": sel})

            return {"model_used": variant, "catalysts": results}

        except Exception as e:
            logger.error(f"CMC predict error: {e}")
            return self._mock_predict(variant, n_candidates)

    # ── Mock ───────────────────────────────────────────────────────────────────

    _MOCK_CATALYSTS = [
        ("CC(=O)Cl",         "[C][C][=O][Cl]"),
        ("c1ccccc1",         "[C][=C][C][=C][C][=C][Ring1][=A]"),
        ("CC#N",             "[C][C][#N]"),
        ("OC(=O)c1ccccc1",  "[O][C][=O][C][=C][C][=C][C][=C][Ring1][=A]"),
        ("c1ccncc1",         "[C][=C][C][=N][C][=C][Ring1][=A]"),
        ("CCOCC",            "[C][C][O][C][C]"),
        ("CC(C)=O",          "[C][C][Branch1][C][C][=O]"),
        ("CC(=O)OCC",        "[C][C][=O][O][C][C]"),
        ("CCCl",             "[C][C][Cl]"),
        ("CBr",              "[C][Br]"),
    ]

    def _mock_predict(self, variant: str, n: int) -> dict:
        sample = random.sample(self._MOCK_CATALYSTS, min(n, len(self._MOCK_CATALYSTS)))
        catalysts = [{"smiles": smi, "selfies": sel} for smi, sel in sample]
        return {"model_used": variant, "catalysts": catalysts, "mock": True}


# ── Singleton ─────────────────────────────────────────────────────────────────
cmc_service = CMCService()
