"""
Cataltys — FastAPI Main Application
=====================================
Startup: loads all AI models lazily in background threads.
CORS:    allows Next.js dev server (localhost:3000).
Routers: /api/catalyst, /api/simulation
"""

import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from routers.catalyst   import router as catalyst_router
from routers.simulation import router as simulation_router
from services.vocab_service   import vocab_service
from services.cmc_service     import cmc_service
from services.ranking_service import ranking_service
from services.reaction_service import reaction_service

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("main")


# ── Lifespan: load models at startup ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Cataltys backend starting ===")

    loop = asyncio.get_event_loop()

    # ── Step 1: load SELFIES vocabulary synchronously ─────────────────────────
    # Must finish before models load so they all share the same stoi/itos tables.
    logger.info("Loading SELFIES vocabulary from training cache…")
    await loop.run_in_executor(None, vocab_service.load)
    summary = vocab_service.summary()
    logger.info(
        f"Vocabulary ready: {summary['vocab_size']} tokens from "
        f"{summary['training_mols']} training molecules "
        f"(cache: {summary['cache_path']})"
    )

    # ── Step 2: load AI models in background ─────────────────────────────────
    async def load_all():
        await loop.run_in_executor(None, cmc_service.load)
        await loop.run_in_executor(None, ranking_service.load)
        await loop.run_in_executor(None, reaction_service.load)

    asyncio.create_task(load_all())
    logger.info("Model loading started in background…")

    yield

    # Cleanup
    logger.info("=== Cataltys backend shutting down ===")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Cataltys API",
    description = "AI-driven catalyst prediction and reaction simulation platform",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log every 422 validation error in detail so we can debug schema mismatches."""
    body = None
    try:
        body = await request.body()
        body = body.decode("utf-8")
    except Exception:
        pass

    errors = exc.errors()
    logger.error(
        f"422 Validation error on {request.method} {request.url.path}\n"
        f"  Body: {body}\n"
        f"  Errors: {errors}"
    )
    return JSONResponse(
        status_code=422,
        content={"detail": errors, "body_received": body},
    )


app.include_router(catalyst_router)
app.include_router(simulation_router)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":    "ok",
        "vocabulary": vocab_service.summary(),
        "models": {
            "cmc_loaded":      cmc_service.loaded,
            "cmc_mock":        cmc_service.mock,
            "cmc_variants":    list(cmc_service.models.keys()),
            "ranking_loaded":  ranking_service.loaded,
            "ranking_mock":    ranking_service.mock,
            "reaction_loaded": reaction_service.loaded,
            "reaction_mock":   reaction_service.mock,
        },
    }


@app.get("/health/vocab")
async def health_vocab():
    """Detailed vocabulary status — useful for debugging."""
    return {
        **vocab_service.summary(),
        "sample_tokens": sorted(vocab_service.alphabet)[:20],
    }


@app.get("/")
async def root():
    return {"message": "Cataltys AI Chemistry Platform — API v1.0"}
