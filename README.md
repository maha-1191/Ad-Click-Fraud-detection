Ad Click Fraud Classification System
About the Project

This project is an Ad Click Fraud Classification System designed to detect fraudulent and legitimate ad clicks in digital advertising platforms.
Online advertising systems are frequently targeted by automated bots and malicious users who generate fake clicks, leading to financial loss and distorted campaign analytics.

To address this problem, the system uses a hybrid deep learning framework that learns complex click behavior patterns and accurately identifies fraud.

Hybrid Learning Approach

The system combines multiple models to improve robustness and accuracy:

CNN for extracting spatial patterns from click features

RNN / LSTM for modeling sequential and temporal click behavior

XGBoost for final fraud probability prediction

To ensure transparency, SHAP explainability is used to interpret model decisions and highlight the most influential behavioral patterns.

Web Application Features

The solution is implemented as a Django-based web application that allows users to:

Upload clickstream CSV files

Run fraud detection locally

View fraud statistics and risk summaries

Analyze IP-level risk and time-based trends

Understand predictions through explainable AI outputs

The system is modular, well-logged, and designed following good software engineering practices.

Key Highlights

Accurate ad click fraud detection

Hybrid model: CNN + RNN/LSTM + XGBoost

Explainable predictions using SHAP

Fully local Django-based execution

Scalable and production-ready design

Local Setup Instructions

Create a virtual environment:

python -m venv venv


Activate the environment:

venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Setup the database:

python manage.py makemigrations
python manage.py migrate


Train the fraud detection model:

python -m fraudapp.ml_engine.training.run_training


Run the Django server:

python manage.py runserver --noreload

Access URLs

Application: http://127.0.0.1:8000/








