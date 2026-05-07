"""
Simulation API Router
======================
REST endpoints for session lifecycle + WebSocket stream for live updates.

POST   /api/simulation/start
POST   /api/simulation/pause
POST   /api/simulation/resume
POST   /api/simulation/stop
POST   /api/simulation/reset
POST   /api/simulation/step
POST   /api/simulation/conditions
GET    /api/simulation/state/{session_id}
WS     /api/simulation/stream/{session_id}
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from services.reaction_service import reaction_service

logger = logging.getLogger("simulation_router")
router = APIRouter(prefix="/api/simulation", tags=["simulation"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class MoleculeInput(BaseModel):
    smiles:   str   = "O"
    selfies:  str   = ""
    name:     str   = "Molecule"
    quantity: float = 1.0

class StartRequest(BaseModel):
    reactants:   list[MoleculeInput] = Field(default=[])
    catalysts:   list[MoleculeInput] = Field(default=[])
    solvents:    list[MoleculeInput] = Field(default=[])
    temperature: float = 298.0
    ph:          float = 7.0
    pressure:    float = 1.0
    volume:      float = 0.5
    stirring:    float = 300.0
    voltage:     float = 0.0
    gas_flow:    float = 0.0
    speed:       float = 1.0
    step_mode:   bool  = False
    container:   str   = "beaker"

class SessionRequest(BaseModel):
    session_id: str

class ConditionsUpdate(BaseModel):
    session_id:  str
    temperature: Optional[float] = None
    ph:          Optional[float] = None
    pressure:    Optional[float] = None
    stirring:    Optional[float] = None
    voltage:     Optional[float] = None
    gas_flow:    Optional[float] = None
    speed:       Optional[float] = None


# ── REST endpoints ─────────────────────────────────────────────────────────────

@router.post("/start")
async def start_simulation(req: StartRequest):
    config = {
        "reactants":   [m.model_dump() for m in req.reactants],
        "catalysts":   [m.model_dump() for m in req.catalysts],
        "solvents":    [m.model_dump() for m in req.solvents],
        "temperature": req.temperature,
        "ph":          req.ph,
        "pressure":    req.pressure,
        "volume":      req.volume,
        "stirring":    req.stirring,
        "voltage":     req.voltage,
        "gas_flow":    req.gas_flow,
        "speed":       req.speed,
        "step_mode":   req.step_mode,
        "container":   req.container,
    }
    session_id = reaction_service.create_session(config)
    reaction_service.start(session_id)
    return {"session_id": session_id, "status": "running"}


@router.post("/pause")
async def pause_simulation(req: SessionRequest):
    sess = reaction_service.get_session(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    reaction_service.pause(req.session_id)
    return {"status": "paused"}


@router.post("/resume")
async def resume_simulation(req: SessionRequest):
    sess = reaction_service.get_session(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    reaction_service.start(req.session_id)
    return {"status": "running"}


@router.post("/stop")
async def stop_simulation(req: SessionRequest):
    reaction_service.stop(req.session_id)
    reaction_service.remove_session(req.session_id)
    return {"status": "stopped"}


@router.post("/reset")
async def reset_simulation(req: SessionRequest):
    sess = reaction_service.get_session(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    reaction_service.reset(req.session_id)
    return {"status": "reset", "state": sess.state_as_dict()}


@router.post("/step")
async def step_simulation(req: SessionRequest):
    """Advance by exactly one step (step-by-step mode)."""
    state = reaction_service.step_once(req.session_id)
    if state is None:
        raise HTTPException(404, "Session not found")
    return state


@router.post("/conditions")
async def update_conditions(req: ConditionsUpdate):
    sess = reaction_service.get_session(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")

    updates = req.model_dump(exclude={"session_id"}, exclude_none=True)
    if "speed" in updates:
        sess.speed = float(updates.pop("speed"))
    if updates:
        reaction_service.update_conditions(req.session_id, updates)

    return {"status": "updated", "conditions": sess.state.conditions.__dict__}


@router.get("/state/{session_id}")
async def get_state(session_id: str):
    sess = reaction_service.get_session(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return sess.state_as_dict()


# ── WebSocket stream ──────────────────────────────────────────────────────────

@router.websocket("/stream/{session_id}")
async def simulation_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    sess = reaction_service.get_session(session_id)

    if not sess:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return

    async def send_update(payload: dict):
        await websocket.send_text(json.dumps(payload))

    sess.add_ws_callback(send_update)
    logger.info(f"WS connected to session {session_id}")

    try:
        # Send initial state immediately
        await websocket.send_text(json.dumps(sess.state_as_dict()))

        # Listen for incoming control messages from the client
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg  = json.loads(data)
                action = msg.get("action")
                if action == "pause":
                    reaction_service.pause(session_id)
                elif action == "resume":
                    reaction_service.start(session_id)
                elif action == "step":
                    state = reaction_service.step_once(session_id)
                    if state:
                        await websocket.send_text(json.dumps(state))
                elif action == "conditions":
                    reaction_service.update_conditions(session_id, msg.get("data", {}))
                    if "speed" in msg.get("data", {}):
                        sess.speed = float(msg["data"]["speed"])
            except asyncio.TimeoutError:
                # Send a ping to keep connection alive
                await websocket.send_json({"ping": True})

    except WebSocketDisconnect:
        logger.info(f"WS disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"WS error [{session_id}]: {e}")
    finally:
        sess.remove_ws_callback(send_update)
