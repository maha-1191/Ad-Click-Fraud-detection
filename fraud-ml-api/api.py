from fastapi import FastAPI, UploadFile, File
import pandas as pd

from ml_engine.inference.predictor import FraudPredictor

app = FastAPI(title="Ad Click Fraud Detection API")

predictor = FraudPredictor()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    return predictor.predict(df)

