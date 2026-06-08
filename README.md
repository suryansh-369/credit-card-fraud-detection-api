# Credit Card Fraud Detection API

## Project Overview

This project builds and deploys a machine learning system for detecting fraudulent credit card transactions using the Credit Card Fraud Detection dataset.

The project covers the complete machine learning workflow:

* Exploratory Data Analysis (EDA)
* Class imbalance analysis
* Model training and evaluation
* Threshold tuning
* Model serialization
* FastAPI deployment
* REST API inference

---

## Dataset

Credit Card Fraud Detection Dataset

Dataset Characteristics:

* Total Transactions: 284,807
* Fraud Transactions: 492
* Fraud Rate: 0.17%
* Features:

  * Time
  * Amount
  * V1–V28 (PCA-transformed features)

The dataset is highly imbalanced, making accuracy an unreliable metric for evaluation.

The original Credit Card Fraud Detection dataset is not included in this repository due to its size (~144 MB).

Users can download the dataset separately and place it in the project directory before running the training script.

---

## Models Evaluated

The following models were tested:

1. Logistic Regression
2. Logistic Regression with Class Weights
3. SMOTE-based Models
4. Random Forest
5. XGBoost

After comparing precision, recall, false positives, and false negatives, Random Forest was selected as the final deployment model.

---

## Final Model

Model:

RandomForestClassifier

Configuration:

* n_estimators = 20
* random_state = 42
* n_jobs = -1

Operational Threshold:

* 0.4

---

## Test Results

Confusion Matrix:

[[56855, 9],
[13, 85]]

Performance Summary:

* False Positives: 9
* False Negatives: 13
* Frauds Detected: 85
* Legitimate Transactions Correctly Identified: 56,855

The threshold was tuned to balance fraud detection performance while minimizing false alarms.

---

## API Endpoints

### Home Endpoint

GET /

Response:

{
"message": "Fraud Detection API is working"
}

### Prediction Endpoint

POST /predict

Input:

{
"Time": 406,
"V1": -2.312227,
"...": "...",
"Amount": 0
}

Output:

{
"fraud_probability": 0.75,
"prediction": 1,
"result": "Fraud"
}

---

## Running Locally

Install dependencies:

pip install -r requirements.txt

Run API:

python -m uvicorn app:app --reload

Open Swagger UI:

http://127.0.0.1:8000/docs

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* FastAPI
* Joblib
* Uvicorn

---

## Future Improvements

* Docker containerization
* CI/CD pipeline
* Model monitoring
* Cloud deployment
* Automated testing

---

## Author

Suryansh

Machine Learning & AI Engineering Project
