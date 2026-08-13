from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.services.credit_scoring_service import (
    CreditScoringService,
)


app = FastAPI(
    title="Credit Scoring Platform",
    description=(
        "Credit risk assessment API using "
        "machine learning, probability calibration, "
        "credit scoring, and automated decisions."
    ),
    version="1.0.0",
)


# ------------------------------------------------------------------
# Load scoring service once when API starts
# ------------------------------------------------------------------

try:
    scoring_service = CreditScoringService()
except Exception as exc:
    scoring_service = None
    startup_error = str(exc)
else:
    startup_error = None


# ------------------------------------------------------------------
# Request schema
# ------------------------------------------------------------------

class CreditApplication(BaseModel):
    features: dict[str, Any]


# ------------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "application": "Credit Scoring Platform",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    if scoring_service is None:
        return {
            "status": "unhealthy",
            "service": "credit_scoring",
            "error": startup_error,
        }

    return {
        "status": "healthy",
        "service": "credit_scoring",
    }


# ------------------------------------------------------------------
# Credit scoring endpoint
# ------------------------------------------------------------------

@app.post("/api/v1/score")
def score_application(
    application: CreditApplication,
):
    if scoring_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Credit scoring service is not "
                "available."
            ),
        )

    try:

        applicant = pd.DataFrame(
            [application.features]
        )

        result = (
            scoring_service.score_applicant(
                applicant
            )
        )

        return {
            "status": "success",
            "assessment": result,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )