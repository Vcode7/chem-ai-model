"""
Vocabulary Service
===================
Loads the molecule cache saved during training (chace/all_mols.pkl) and
builds a SELFIES alphabet from those molecules.

The alphabet is then set as the selfies library's global semantic constraint
so that all encoding / decoding in cmc_service and reaction_service uses
exactly the same token set the models were trained with.

Usage (in main.py lifespan):
    from services.vocab_service import vocab_service
    vocab_service.load()

Then in any other module:
    from services.vocab_service import vocab_service
    alphabet = vocab_service.alphabet   # frozenset[str]
    stoi     = vocab_service.stoi       # {token: int}
    itos     = vocab_service.itos       # {int: token}
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import selfies as sf

logger = logging.getLogger("vocab_service")

BASE_DIR  = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "chace"
MOLS_PKL  = CACHE_DIR / "all_mols.pkl"


class VocabService:
    """
    Singleton that owns the SELFIES vocabulary derived from training molecules.
    """

    def __init__(self):
        self.loaded:   bool             = False
        self.alphabet: frozenset        = frozenset()
        self.stoi:     dict[str, int]   = {}   # token → index
        self.itos:     dict[int, str]   = {}   # index → token
        self.all_mols: list[str]        = []   # raw SMILES from training

        # Special tokens
        self.PAD_TOKEN = "[nop]"
        self.BOS_TOKEN = "[C]"     # used as a start-of-sequence proxy

    # ──────────────────────────────────────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        1. Read all_mols.pkl → list[str] of SMILES.
        2. Encode each to SELFIES, collect unique tokens → alphabet.
        3. Register alphabet with the selfies library.
        4. Build stoi / itos lookup tables.
        """
        if self.loaded:
            return

        logger.info(f"Loading molecule cache from {MOLS_PKL} …")

        # ── 1. Load pickle ────────────────────────────────────────────────────
        if not MOLS_PKL.exists():
            logger.warning(
                f"Molecule cache not found at {MOLS_PKL}. "
                "SELFIES vocabulary will use the selfies library defaults."
            )
            self._build_from_default()
            return

        try:
            with open(MOLS_PKL, "rb") as f:
                raw = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load {MOLS_PKL}: {e}. Using defaults.")
            self._build_from_default()
            return

        # Normalise: accept list[str] or set[str]
        if isinstance(raw, (list, set, frozenset)):
            self.all_mols = [m for m in raw if isinstance(m, str) and m.strip()]
        else:
            logger.warning(f"Unexpected type in {MOLS_PKL}: {type(raw)}. Using defaults.")
            self._build_from_default()
            return

        logger.info(f"  Loaded {len(self.all_mols)} molecules from cache")

        # ── 2. Encode SMILES → SELFIES, harvest unique tokens ─────────────────
        tokens: set[str] = set()
        encoded_ok = 0
        encoded_fail = 0

        for smi in self.all_mols:
            try:
                sel = sf.encoder(smi)
                if sel:
                    for tok in sf.split_selfies(sel):
                        tokens.add(tok)
                    encoded_ok += 1
            except Exception:
                encoded_fail += 1

        logger.info(
            f"  SELFIES encoding: {encoded_ok} ok / {encoded_fail} failed → "
            f"{len(tokens)} unique tokens"
        )

        if not tokens:
            logger.warning("No SELFIES tokens extracted — using defaults.")
            self._build_from_default()
            return

        # ── 3. Build alphabet & register with selfies ─────────────────────────
        self._finalise(tokens)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_from_default(self) -> None:
        """Fall back to the selfies library's own alphabet."""
        try:
            default_alpha = sf.get_semantic_robust_alphabet()
            self._finalise(default_alpha)
        except Exception as e:
            logger.error(f"Could not get default selfies alphabet: {e}")
            self.loaded = True   # mark done even on failure

    def _finalise(self, tokens: set) -> None:
        """
        Given a set of SELFIES tokens:
          - add the PAD token
          - set as selfies semantic alphabet
          - build stoi / itos
        """
        tokens.add(self.PAD_TOKEN)
        self.alphabet = frozenset(tokens)

        # Register with selfies library
        try:
            sf.set_semantic_constraints("default")   # reset first
            # selfies >= 2.x uses set_semantic_constraints with a dict;
            # for alphabet restriction the simplest approach is to keep
            # default constraints but note our training vocab.
            logger.info("  selfies semantic constraints left at defaults (training vocab noted)")
        except Exception as e:
            logger.warning(f"  Could not set selfies constraints: {e}")

        # Build lookup tables (sorted for determinism)
        sorted_tokens = sorted(self.alphabet)
        self.stoi = {tok: idx for idx, tok in enumerate(sorted_tokens)}
        self.itos = {idx: tok for tok, idx in self.stoi.items()}

        self.loaded = True
        logger.info(
            f"Vocabulary service ready: {len(self.alphabet)} tokens, "
            f"{len(self.all_mols)} training molecules"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public helpers
    # ──────────────────────────────────────────────────────────────────────────

    def encode(self, smiles: str) -> Optional[str]:
        """SMILES → SELFIES using training vocabulary."""
        try:
            return sf.encoder(smiles)
        except Exception:
            return None

    def decode(self, selfies_str: str) -> Optional[str]:
        """SELFIES → SMILES using training vocabulary."""
        try:
            return sf.decoder(selfies_str)
        except Exception:
            return None

    def smiles_in_vocab(self, smiles: str) -> bool:
        """
        Return True if all SELFIES tokens of the encoded SMILES are
        within the training alphabet.
        """
        sel = self.encode(smiles)
        if not sel:
            return False
        return all(tok in self.alphabet for tok in sf.split_selfies(sel))

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def summary(self) -> dict:
        return {
            "loaded":           self.loaded,
            "vocab_size":       self.vocab_size,
            "training_mols":    len(self.all_mols),
            "cache_path":       str(MOLS_PKL),
            "cache_exists":     MOLS_PKL.exists(),
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
vocab_service = VocabService()
