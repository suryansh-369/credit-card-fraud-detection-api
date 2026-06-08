import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load('random_forest_pipeline.pkl')
THRESHOLD = 0.4
class InputData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API is working"}

@app.post("/predict")
def predict(transaction: InputData):

    data = pd.DataFrame([transaction.model_dump()])

    probability = model.predict_proba(data)[0][1]

    prediction = int(probability >= THRESHOLD)

    return {
        "fraud_probability": round(float(probability), 4)  ,
        "prediction": prediction,
        "result": "Fraud" if prediction else "Legitimate"
    }