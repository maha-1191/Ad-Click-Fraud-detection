from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import pandas as pd
import traceback

from ml_engine.inference.predictor import FraudPredictor

# =====================================================
# APP SETUP
# =====================================================
app = FastAPI(
    title="Ad Click Fraud Detection API",
    description="ML inference API for Ad Click Fraud Detection",
    version="1.0.0"
)

# =====================================================
# CORS
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD MODEL ONCE
# =====================================================
try:
    predictor = FraudPredictor()
except Exception as e:
    raise RuntimeError(f"Failed to load ML model: {e}")

# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

# =====================================================
# PREDICTION ENDPOINT
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    try:
        df = pd.read_csv(file.file)

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        result = predictor.predict(df)

        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Invalid ML output")

        return result

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

