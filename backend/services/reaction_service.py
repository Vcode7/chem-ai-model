"""
Reaction Simulation Service
============================
Wraps the ReactionWorldModel from reaction.py.
Manages per-session simulation state and streams updates via WebSocket.

Session lifecycle:
  start  → create SimSession, begin background step loop
  pause  → suspend step loop
  resume → resume step loop
  stop   → terminate and remove session
  reset  → re-initialise session state
"""

import asyncio
import logging
import random
import time
import uuid
import importlib.util
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np

logger = logging.getLogger("reaction_service")

BASE_DIR       = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "reaction"
MODELS_DIR     = BASE_DIR / "models"


def _import_reaction():
    spec = importlib.util.spec_from_file_location("reaction", MODELS_DIR / "reaction.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Simulation session data ───────────────────────────────────────────────────

@dataclass
class SimMolecule:
    smiles:   str
    selfies:  str
    name:     str
    quantity: float
    role:     str      # reactant | product | intermediate | byproduct
    phase:    str      # liquid | solid | gas | aqueous
    color:    str      # hex color for UI

@dataclass
class SimConditions:
    temperature: float   # K
    ph:          float
    pressure:    float   # atm
    volume:      float   # L
    stirring:    float   # RPM 0-1000
    voltage:     float   # V
    gas_flow:    float   # L/min

@dataclass
class SimState:
    step:        int
    time:        float   # seconds
    molecules:   list[SimMolecule]
    conditions:  SimConditions
    events:      list[str]
    reaction_velocity: float
    catalyst_activity: float
    energy_history:    list[float]
    ph_history:        list[float]
    temperature_history: list[float]


MOLECULE_COLORS = [
    "#00E5FF", "#7B2FBE", "#00FF88", "#FF6B35",
    "#F7C948", "#FF4B7B", "#4BFFDE", "#A78BFA",
]

def _assign_color(idx: int) -> str:
    return MOLECULE_COLORS[idx % len(MOLECULE_COLORS)]


def _make_initial_state(config: dict) -> SimState:
    """Build starting SimState from user config dict."""
    mols = []
    for i, r in enumerate(config.get("reactants", [])):
        mols.append(SimMolecule(
            smiles   = r.get("smiles", "O"),
            selfies  = r.get("selfies", ""),
            name     = r.get("name", f"R{i+1}"),
            quantity = float(r.get("quantity", 1.0)),
            role     = "reactant",
            phase    = "liquid",
            color    = _assign_color(i),
        ))
    for i, cat in enumerate(config.get("catalysts", [])):
        mols.append(SimMolecule(
            smiles   = cat.get("smiles", "c1ccccc1"),
            selfies  = cat.get("selfies", ""),
            name     = cat.get("name", f"Cat{i+1}"),
            quantity = float(cat.get("quantity", 0.1)),
            role     = "reactant",
            phase    = "liquid",
            color    = _assign_color(i + 10),
        ))

    cond = SimConditions(
        temperature = float(config.get("temperature", 298.0)),
        ph          = float(config.get("ph", 7.0)),
        pressure    = float(config.get("pressure", 1.0)),
        volume      = float(config.get("volume", 0.5)),
        stirring    = float(config.get("stirring", 300.0)),
        voltage     = float(config.get("voltage", 0.0)),
        gas_flow    = float(config.get("gas_flow", 0.0)),
    )

    return SimState(
        step        = 0,
        time        = 0.0,
        molecules   = mols,
        conditions  = cond,
        events      = ["Simulation initialised."],
        reaction_velocity   = 0.0,
        catalyst_activity   = 0.5,
        energy_history      = [],
        ph_history          = [cond.ph],
        temperature_history = [cond.temperature],
    )


class SimSession:
    def __init__(self, session_id: str, config: dict):
        self.session_id   = session_id
        self.config       = config
        self.state        = _make_initial_state(config)
        self.status       = "paused"  # running | paused | stopped
        self.speed        = float(config.get("speed", 1.0))
        self.step_mode    = bool(config.get("step_mode", False))
        self.ws_callbacks = []   # list of async callables
        self._task: Optional[asyncio.Task] = None

    def add_ws_callback(self, cb):
        self.ws_callbacks.append(cb)

    def remove_ws_callback(self, cb):
        self.ws_callbacks = [c for c in self.ws_callbacks if c is not cb]

    def state_as_dict(self) -> dict:
        s = self.state
        return {
            "session_id": self.session_id,
            "status":     self.status,
            "step":       s.step,
            "time":       s.time,
            "conditions": asdict(s.conditions),
            "molecules":  [asdict(m) for m in s.molecules],
            "events":     s.events[-20:],
            "reaction_velocity":     s.reaction_velocity,
            "catalyst_activity":     s.catalyst_activity,
            "energy_history":        s.energy_history[-60:],
            "ph_history":            s.ph_history[-60:],
            "temperature_history":   s.temperature_history[-60:],
        }


# ── Deterministic step logic (mirrors reaction.py heuristics) ─────────────────

_ACID_SMILES = {"CC(=O)O", "CC(O)=O", "OC(=O)c1ccccc1", "OS(=O)(=O)O", "OC(=O)C=O"}
_BASE_SMILES = {"N", "CCN(CC)CC", "c1ccc(N)cc1", "[OH-]"}

def _evolve_step(state: SimState, dt: float = 5.0) -> SimState:
    """Advance simulation by dt seconds using deterministic chemistry rules."""
    cond = state.conditions
    mols = list(state.molecules)
    events = []

    # Arrhenius rate constant
    A, Ea_R = 0.01, 3000.0
    cat_act  = max((m.quantity for m in mols if "cat" in m.name.lower()), default=0.1)
    k        = A * np.exp(-Ea_R / max(cond.temperature, 200.0)) * (1.0 + cat_act * 5.0)
    k       *= (cond.stirring / 300.0 + 0.5)  # stirring boosts rate

    # Reactant consumption → product formation
    new_mols = []
    product_gain = 0.0
    for m in mols:
        if m.role == "reactant" and m.quantity > 0:
            consumed = min(m.quantity, k * m.quantity * dt)
            product_gain += consumed * 0.9
            new_m = SimMolecule(**{**asdict(m), "quantity": round(max(0.0, m.quantity - consumed), 4)})
            if consumed > 0.001:
                events.append(f"⚡ {m.name} consumed {consumed:.3f} mol")
            new_mols.append(new_m)
        elif m.role == "product":
            new_q = round(m.quantity + product_gain / max(len([x for x in mols if x.role == "product"]), 1), 4)
            new_mols.append(SimMolecule(**{**asdict(m), "quantity": new_q}))
            product_gain = 0.0
        else:
            new_mols.append(m)

    # If product_gain is leftover (no products yet), spawn one
    if product_gain > 0 and not any(m.role == "product" for m in new_mols):
        new_mols.append(SimMolecule(
            smiles="CC(=O)O", selfies="[C][C][=O][O]",
            name="Product", quantity=round(product_gain, 4),
            role="product", phase="liquid", color="#00FF88",
        ))
        events.append("✨ Product appeared")

    # Acid-base neutralisation
    acids = [m for m in new_mols if m.smiles in _ACID_SMILES and m.role == "reactant"]
    bases = [m for m in new_mols if m.smiles in _BASE_SMILES and m.role == "reactant"]
    new_ph = cond.ph
    if acids and bases:
        rate = 0.002 * dt
        transferred = min(acids[0].quantity * rate, bases[0].quantity * rate, 0.5)
        if new_ph < 7:
            new_ph = min(7.0, new_ph + transferred * 2.0)
        else:
            new_ph = max(7.0, new_ph - transferred * 2.0)
        events.append(f"🧪 Neutralisation: pH → {new_ph:.2f}")

    # Temperature fluctuation
    dT    = (random.random() - 0.5) * 1.5
    new_T = max(200.0, cond.temperature + dT)

    new_cond = SimConditions(
        temperature = round(new_T, 2),
        ph          = round(min(14.0, max(0.0, new_ph)), 3),
        pressure    = round(max(0.1, cond.pressure + (random.random() - 0.5) * 0.02), 3),
        volume      = cond.volume,
        stirring    = cond.stirring,
        voltage     = cond.voltage,
        gas_flow    = cond.gas_flow,
    )

    # Reaction velocity
    total_reactant = sum(m.quantity for m in new_mols if m.role == "reactant")
    v = round(k * total_reactant, 5)

    energy = round(-abs(np.random.normal(1.0, 0.1)), 4)

    new_ph_hist = state.ph_history + [new_cond.ph]
    new_T_hist  = state.temperature_history + [new_cond.temperature]
    new_e_hist  = state.energy_history + [energy]

    return SimState(
        step        = state.step + 1,
        time        = round(state.time + dt, 2),
        molecules   = new_mols,
        conditions  = new_cond,
        events      = state.events + events,
        reaction_velocity   = v,
        catalyst_activity   = round(cat_act * 0.999, 4),  # slow deactivation
        energy_history      = new_e_hist,
        ph_history          = new_ph_hist,
        temperature_history = new_T_hist,
    )


# ── Service ───────────────────────────────────────────────────────────────────

class ReactionService:
    def __init__(self):
        self.sessions: Dict[str, SimSession] = {}
        self.reaction_mod = None
        self.model        = None
        self.loaded       = False
        self.mock         = True
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        """Load reaction world model checkpoint."""
        logger.info("Loading Reaction World Model…")
        checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"))
        if not checkpoints:
            logger.warning("No reaction checkpoint — using deterministic mock")
            self.mock   = True
            self.loaded = True
            return

        ckpt_path = checkpoints[-1]
        try:
            self.reaction_mod = _import_reaction()
            self.model = self.reaction_mod.ReactionWorldModel().to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.mock = False
            logger.info(f"Reaction model loaded from {ckpt_path.name} ✓")
        except Exception as e:
            logger.error(f"Reaction model load error: {e} — using deterministic mock")
            self.mock = True

        self.loaded = True

    # ── Session management ─────────────────────────────────────────────────────

    def create_session(self, config: dict) -> str:
        if not self.loaded:
            self.load()
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = SimSession(session_id, config)
        logger.info(f"Session created: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[SimSession]:
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str):
        sess = self.sessions.pop(session_id, None)
        if sess and sess._task:
            sess._task.cancel()

    def start(self, session_id: str):
        sess = self.sessions.get(session_id)
        if not sess:
            return
        sess.status = "running"
        if sess._task is None or sess._task.done():
            sess._task = asyncio.create_task(self._run_loop(sess))

    def pause(self, session_id: str):
        sess = self.sessions.get(session_id)
        if sess:
            sess.status = "paused"

    def stop(self, session_id: str):
        sess = self.sessions.get(session_id)
        if sess:
            sess.status = "stopped"
            if sess._task:
                sess._task.cancel()

    def reset(self, session_id: str):
        sess = self.sessions.get(session_id)
        if not sess:
            return
        if sess._task:
            sess._task.cancel()
        sess.state  = _make_initial_state(sess.config)
        sess.status = "paused"
        sess._task  = None

    def update_conditions(self, session_id: str, conditions: dict):
        sess = self.sessions.get(session_id)
        if not sess:
            return
        c = sess.state.conditions
        for k, v in conditions.items():
            if hasattr(c, k):
                setattr(c, k, float(v))

    def step_once(self, session_id: str) -> Optional[dict]:
        sess = self.sessions.get(session_id)
        if not sess:
            return None
        sess.state = _evolve_step(sess.state)
        return sess.state_as_dict()

    # ── Background loop ────────────────────────────────────────────────────────

    async def _run_loop(self, sess: SimSession):
        while sess.status == "running":
            try:
                sess.state = _evolve_step(sess.state)
                payload = sess.state_as_dict()

                # Broadcast to all connected WebSocket callbacks
                dead = []
                for cb in sess.ws_callbacks:
                    try:
                        await cb(payload)
                    except Exception:
                        dead.append(cb)
                for d in dead:
                    sess.ws_callbacks.remove(d)

                # Step delay based on speed
                delay = max(0.1, 1.0 / sess.speed)
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Step loop error [{sess.session_id}]: {e}")
                await asyncio.sleep(1.0)


# ── Singleton ─────────────────────────────────────────────────────────────────
reaction_service = ReactionService()
