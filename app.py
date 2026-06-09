import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI 
import joblib
from fastapi import UploadFile, File 
from fastapi.responses import FileResponse
import uuid

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

@app.post("/batch_predict")
def batch_predict(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    
    if "Class" in data.columns:
        data = data.drop("Class", axis=1)
    
    expexted_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    missing_cols = set(expexted_columns) - set(data.columns)
    
    if missing_cols:
        return {"error": "Missing columns",
                "Missing Column": missing_cols}

    invalid_columns = []
    for col in expexted_columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except:
            invalid_columns.append(col)
    if invalid_columns:
        return {
            "error": "Invalid numeric columns",
            "invalid_columns": invalid_columns
            }
    
    
    probabilities = model.predict_proba(data)[:, 1]
    predictions = (probabilities >= THRESHOLD)

    data['fraud_probability'] = probabilities.round(4)
    data['prediction'] = predictions.astype(int)

    
    #fraud_predictions = (data["prediction"] == 1).sum()
    #legitimate_predictions = (data["prediction"] == 0).sum()
    
    

    data["prediction_label"] = data["prediction"].map({0: "Legitimate",1: "Fraud"})    
    
    filename = f"predictions_{uuid.uuid4()}.csv"
    data.to_csv(filename, index=False)
    
    return FileResponse(
        filename,
        media_type="text/csv",
        filename="predictions.csv"
    )
    
  