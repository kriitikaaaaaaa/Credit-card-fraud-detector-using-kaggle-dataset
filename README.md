# Credit Card Fraud Detection

A machine-learning-based web application for analyzing credit card transactions and identifying potentially fraudulent activity.

## 🚀 Live Demo

**Streamlit App:**  
https://credit-card-fraud-detector-using-kaggle-dataset-vnyf7ntl5lsv8j.streamlit.app/

## 📌 Overview

This project uses an anonymized credit card transaction dataset to demonstrate fraud detection with machine learning and a CNN-based classification model.

The Streamlit application provides:

- Single-transaction fraud analysis
- Batch CSV prediction
- Fraud/legitimate classification
- Model confidence scores
- Transaction feature inspection
- Downloadable batch prediction results

## 📊 Dataset

The project is based on the Credit Card Fraud Detection dataset available on Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains transaction features including:

- Time
- Amount
- V1–V28 anonymized features
- Class label

The dataset contains approximately 284,000 transactions, with fraudulent transactions representing a small minority of the data.

## 🧠 Model

The deployed application uses a CNN/Conv1D neural network.

### Input Features

The model uses 30 input features:

- Scaled Time
- Scaled Amount
- V1–V28

### Architecture

1. Conv1D
2. Conv1D
3. Conv1D
4. Flatten
5. Dense sigmoid output

A prediction score of **0.5 or higher** is classified as fraudulent; scores below 0.5 are classified as legitimate.

## 🖥️ Application

The Streamlit interface contains three main sections:

### Single Transaction

Select a transaction from the demonstration dataset and run the trained model to receive a fraud/legitimate prediction.

### Batch Prediction

Upload a CSV containing:

`Time`, `Amount`, and `V1`–`V28`

The application generates predictions and allows the results to be downloaded as a CSV file.

### About the Model

Provides information about the dataset, features, model architecture, and classification method.

## 🔌 REST API

The repository also contains a small Flask-based REST API demonstration with endpoints for adding, retrieving, deleting, and updating results.

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Streamlit
- Flask
- Docker

## 📁 Project Structure

```text
├── API/
├── API.py
├── API_TEST.py
├── ML.py
├── app.py
├── best_model.h5
├── demo_transactions.csv
├── prepare_deployment.py
├── requirements.txt
├── scalers.joblib
├── Dockerfile
└── README.md
