# Credit Card Fraud Detection API

An end-to-end Machine Learning project that detects fraudulent credit card transactions using a Random Forest classifier. The project includes model training, REST API deployment with FastAPI, a Streamlit frontend for user interaction, Docker containerization, and cloud deployment on Render.

## Features

- Single transaction fraud prediction
- Batch fraud prediction using CSV uploads
- Fraud probability scoring
- Downloadable prediction results
- FastAPI REST API
- Streamlit web interface
- Dockerized deployment
- Hosted on Render

## Tech Stack

- Python
- Scikit-learn
- Pandas
- FastAPI
- Streamlit
- Docker
- Render
- Git & GitHub

## Project Architecture

```text
User
  ↓
Streamlit Frontend
  ↓
FastAPI Backend
  ↓
Random Forest Model (.pkl)
  ↓
Fraud Prediction Results
```

## Live Demo

Frontend:
[Your Streamlit URL]

API Documentation:
[Your FastAPI URL]/docs

## API Endpoints

### Health Check
```
GET /
```

### Single Prediction
```
POST /predict
```

### Batch Prediction
```
POST /batch_predict
```

Upload a CSV file and receive a prediction file containing:

- fraud_probability
- prediction
- prediction_label

## Repository

GitHub:
https://github.com/suryansh-369/credit-card-fraud-detection-api

## Author

Suryansh