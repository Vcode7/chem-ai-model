"""
Catalyst API Router
====================
POST /api/catalyst/predict  — predict candidate catalysts (CMC model)
POST /api/catalyst/rank     — rank candidates by energy (OC-Ranking model)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from services.cmc_service     import cmc_service
from services.ranking_service import ranking_service

router = APIRouter(prefix="/api/catalyst", tags=["catalyst"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    reactants:    list[str] = Field(..., description="List of reactant strings (SMILES or SELFIES)")
    solvents:     list[str] = Field(default=[], description="Optional solvent strings")
    products:     list[str] = Field(default=[], description="Optional product strings")
    input_format: str       = Field(default="smiles", description="'smiles' or 'selfies'")
    n_candidates: int       = Field(default=8, ge=1, le=20)
    conf_threshold: float   = Field(default=0.0, ge=-10.0, le=10.0)

class RankRequest(BaseModel):
    reactants:    list[str] = Field(..., description="Reactant SMILES list")
    catalysts:    list[dict] = Field(..., description="List of {smiles, selfies} dicts")
    input_format: str        = Field(default="smiles")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict_catalysts(req: PredictRequest):
    """
    Predict candidate catalysts using CMC models A/B/C.
    Auto-selects model based on which inputs are provided.
    """
    if not req.reactants:
        raise HTTPException(status_code=422, detail="At least one reactant is required.")

    if req.input_format not in ("smiles", "selfies"):
        raise HTTPException(status_code=422, detail="input_format must be 'smiles' or 'selfies'.")

    result = cmc_service.predict(
        reactants    = req.reactants,
        solvents     = req.solvents,
        products     = req.products,
        input_format = req.input_format,
        conf_threshold = req.conf_threshold,
        n_candidates = req.n_candidates,
    )

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@router.post("/rank")
async def rank_catalysts(req: RankRequest):
    """
    Score each catalyst against every reactant using the OC-Ranking model.
    Returns results sorted by average adsorption energy (lower = better).
    """
    if not req.reactants:
        raise HTTPException(status_code=422, detail="At least one reactant is required.")
    if not req.catalysts:
        raise HTTPException(status_code=422, detail="No catalysts provided.")

    results = ranking_service.rank(
        reactant_smiles_list = req.reactants,
        catalyst_list        = req.catalysts,
        input_format         = req.input_format,
    )
    return {"rankings": results, "count": len(results)}


@router.get("/status")
async def catalyst_service_status():
    """Health-check for model loading status."""
    return {
        "cmc":  {
            "loaded":   cmc_service.loaded,
            "mock":     cmc_service.mock,
            "variants": list(cmc_service.models.keys()),
        },
        "ranking": {
            "loaded": ranking_service.loaded,
            "mock":   ranking_service.mock,
        },
    }
