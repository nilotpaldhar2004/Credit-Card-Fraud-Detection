"""
Credit Card Fraud Detection API
================================
A production-grade FastAPI service for real-time transaction fraud scoring
using a pre-trained LightGBM model.

Author  : Nilotpal Dhar
Version : 2.0.0
"""

import os
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraud-api")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "lightgbm_fraud_model.pkl")
HOST       = os.getenv("HOST", "0.0.0.0")
PORT       = int(os.getenv("PORT", 8500))
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", 0.5))

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

# ---------------------------------------------------------------------------
# Pydantic schema – strict validation keeps bad payloads out of the model
# ---------------------------------------------------------------------------
class TransactionRequest(BaseModel):
    """
    Represents a single credit-card transaction.
    All 30 numeric features are required (Time, V1-V28, Amount).
    """

    Time:   float = Field(..., description="Seconds elapsed since first transaction in the dataset")
    V1:     float; V2:  float; V3:  float; V4:  float; V5:  float
    V6:     float; V7:  float; V8:  float; V9:  float; V10: float
    V11:    float; V12: float; V13: float; V14: float; V15: float
    V16:    float; V17: float; V18: float; V19: float; V20: float
    V21:    float; V22: float; V23: float; V24: float; V25: float
    V26:    float; V27: float; V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount in USD (must be ≥ 0)")

    @field_validator("Amount")
    @classmethod
    def amount_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Amount must be a non-negative number.")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "Time": 1205.0,
            "V1": -0.8, "V2": 0.5, "V3": 1.2, "V4": 0.1, "V5": -0.3,
            "V6": 0.0,  "V7": 0.8, "V8": -0.1,"V9": 0.4, "V10": -0.2,
            "V11": 0.5, "V12": 0.1,"V13": -0.5,"V14": -1.2,"V15": 0.3,
            "V16": 0.4, "V17": 0.8,"V18": -0.2,"V19": 0.5, "V20": 0.1,
            "V21": 0.0, "V22": 0.1,"V23": -0.2,"V24": 0.3, "V25": 0.1,
            "V26": -0.1,"V27": 0.0,"V28": 0.0, "Amount": 250.00,
        }
    }}


class PredictionResponse(BaseModel):
    status:            str
    fraud_probability: float
    is_fraud:          bool
    action:            str
    latency_ms:        float


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    version:      str = "2.0.0"


# ---------------------------------------------------------------------------
# Application state (holds shared objects across requests)
# ---------------------------------------------------------------------------
class AppState:
    model: Any = None
    model_loaded: bool = False


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan – load model at startup, release at shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once when the server starts."""
    logger.info("Loading model from '%s' …", MODEL_PATH)
    try:
        app_state.model = joblib.load(MODEL_PATH)
        app_state.model_loaded = True
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.error(
            "Model file '%s' not found. "
            "Place it in the same directory as app.py and restart.",
            MODEL_PATH,
        )
    yield
    logger.info("Server shutting down. Releasing resources.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Real-time fraud scoring powered by LightGBM. "
        "Submit a transaction's 30-feature vector and receive an immediate "
        "APPROVE / BLOCK decision with a probability score."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to specific origins in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the dashboard HTML file."""
    if not os.path.exists("index.html"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="index.html not found. Ensure it lives beside app.py.",
        )
    return FileResponse("index.html")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Liveness & readiness probe.
    Returns 503 when the model has not been loaded successfully.
    """
    if not app_state.model_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "degraded", "model_loaded": False, "version": "2.0.0"},
        )
    return HealthResponse(status="ok", model_loaded=True)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Score a transaction for fraud",
)
async def predict_fraud(transaction: TransactionRequest):
    """
    Accepts a validated transaction payload and returns:

    - **fraud_probability** – model's raw probability (0–1)
    - **is_fraud** – boolean decision at the configured threshold
    - **action** – `APPROVE` or `BLOCK`
    - **latency_ms** – inference time in milliseconds
    """
    if not app_state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check server logs.",
        )

    try:
        t0 = time.perf_counter()

        # Build a correctly ordered DataFrame (feature order matters for tree models)
        df = pd.DataFrame([transaction.model_dump()])[FEATURE_COLUMNS]

        fraud_probability = float(app_state.model.predict(df)[0])
        is_fraud          = fraud_probability > FRAUD_THRESHOLD
        latency_ms        = (time.perf_counter() - t0) * 1000

        logger.info(
            "Prediction complete | prob=%.4f | action=%s | latency=%.2fms",
            fraud_probability,
            "BLOCK" if is_fraud else "APPROVE",
            latency_ms,
        )

        return PredictionResponse(
            status="success",
            fraud_probability=round(fraud_probability, 4),
            is_fraud=is_fraud,
            action="BLOCK" if is_fraud else "APPROVE",
            latency_ms=round(latency_ms, 2),
        )

    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting server on %s:%d", HOST, PORT)
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info",
    )