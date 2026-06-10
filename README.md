#  Credit Card Fraud Detection System

An end-to-end Machine Learning application that detects fraudulent credit card transactions in real time using a Random Forest classifier.

The project covers the complete ML lifecycle: data preprocessing, model training, threshold tuning, API development, frontend integration, containerization, and cloud deployment.

---

## 🌐 Live Demo

**Frontend Application**
[Add Your Streamlit URL Here]

**API Documentation**
https://credit-card-fraud-detection-api-1-yba7.onrender.com/docs

**GitHub Repository**
https://github.com/suryansh-369/credit-card-fraud-detection-api

---

## Business Problem

Credit card fraud causes billions of dollars in losses every year. Fraudulent transactions are extremely rare compared to legitimate transactions, making fraud detection a highly imbalanced classification problem.

The goal of this project is to identify fraudulent transactions while minimizing false alarms that could impact genuine customers.

---

## Solution Architecture

```text
User
  ↓
Streamlit Frontend
  ↓
FastAPI Backend
  ↓
Random Forest Model
  ↓
Fraud Prediction
```

---

## Features

* Real-time fraud prediction
* Batch fraud detection via CSV upload
* Fraud probability scoring
* Downloadable prediction results
* REST API built with FastAPI
* Interactive frontend built with Streamlit
* Dockerized application
* Cloud deployment on Render
* Production-ready model serving

---

## Dataset

**Credit Card Fraud Detection Dataset**

* 284,807 transactions
* 492 fraudulent transactions
* Fraud rate: 0.17%
* Highly imbalanced classification problem

Features include:

* Time
* Amount
* V1–V28 (PCA-transformed variables)

---

## Model Development

Models evaluated:

* Logistic Regression
* Weighted Logistic Regression
* SMOTE-based Models
* Random Forest
* XGBoost

After comparing precision, recall, false positives, and false negatives, Random Forest was selected as the final deployment model.

### Final Model

```python
RandomForestClassifier(
    n_estimators=20,
    random_state=42,
    n_jobs=-1
)
```

Threshold tuned to:

```python
0.4
```

to balance fraud detection and false alarms.

---

## Performance

Confusion Matrix:

```text
[[56855, 9],
 [13, 85]]
```

Results:

* Legitimate Transactions Correctly Identified: 56,855
* Fraudulent Transactions Detected: 85
* False Positives: 9
* False Negatives: 13

---

## API Endpoints

### Health Check

```http
GET /
```

### Single Prediction

```http
POST /predict
```

Returns:

```json
{
  "fraud_probability": 0.75,
  "prediction": 1,
  "result": "Fraud"
}
```

### Batch Prediction

```http
POST /batch_predict
```

Upload a CSV file and receive a downloadable prediction file containing:

* fraud_probability
* prediction
* prediction_label

---

## Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit

### Deployment

* Docker
* Render

### Version Control

* Git
* GitHub

---

## 📂 Project Structure

```text
├── app.py
├── frontend.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── random_forest_pipeline.pkl
├── README.md
└── notebook.ipynb
```

---

## 🏃 Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start API:

```bash
python -m uvicorn app:app --reload
```

Start Frontend:

```bash
python -m streamlit run frontend.py
```

---

## 🔮 Future Improvements

* Automated retraining pipeline
* Model monitoring and drift detection
* CI/CD integration
* Cloud-native deployment on AWS
* Experiment tracking with MLflow

---

## 👨‍💻 Author

**Suryansh**

Machine Learning | MLOps | AI Engineering

---

## ❤️ Acknowledgements

Built using open-source technologies and the support of the Python, Scikit-learn, FastAPI, Streamlit, Docker, and ChatGPT communities.
