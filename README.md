# Credit Card Fraud Detection ML App

This project is a Machine Learning–based Credit Card Fraud Detection system deployed as a Streamlit application. The system analyzes transaction details and predicts whether a transaction is fraudulent or legitimate in real time.

# Features

The dataset is highly imbalanced, containing 29,372 legitimate transactions and 628 fraudulent transactions.

Detects fraudulent transactions using a trained machine learning model.

Data preprocessing and feature scaling are implemented using saved files:

scaler_RFC.pkl

label_encoder_RFC.pkl

The SMOTE (Synthetic Minority Over-sampling Technique) method is applied to handle class imbalance and improve prediction performance.

Interactive and user-friendly interface built with Streamlit.

# Model Evaluation

The best model is selected not solely based on accuracy, but by evaluating additional performance metrics such as:

Precision

Recall

F1-score

These metrics provide a better evaluation for imbalanced classification problems.

# Model Saving

The best-performing model, along with the label encoder and feature scaler, is saved using Joblib for future predictions.

# Technologies Used

Python

Pandas

NumPy

Scikit-learn

Streamlit
