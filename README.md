🚨 Ad Click Fraud Classification System
An Advanced Hybrid Deep Learning Framework for Fraud Detection in Digital Advertising Ecosystems
📌 Project Overview

Online digital advertising platforms are increasingly vulnerable to ad click fraud, where malicious bots or deceptive users generate illegitimate clicks to manipulate advertising costs and campaign performance. This project presents a robust, scalable, and explainable web-based system for detecting fraudulent ad clicks using an Advanced Hybrid Deep Learning Technique (HDLT).

The system integrates:

Convolutional Neural Networks (CNN) for spatial feature extraction

Recurrent Neural Networks (RNN/LSTM) for temporal behavior modeling

XGBoost for final fraud classification

SHAP & LIME for explainability and transparency

The application is implemented as a production-ready Django web application with a professional UI, secure backend, and modular ML engine.

🎯 Objectives

Detect fraudulent ad clicks from large-scale clickstream datasets

Combine spatial, temporal, and ensemble learning techniques

Provide interpretable AI predictions using SHAP and LIME

Deliver a real-world usable system, not a lab prototype

Ensure scalability, robustness, and error handling

🧠 System Architecture (Conceptual)

User → Web Interface → Django Backend → ML Engine → Explainability → Dashboard

Dataset upload & validation

Feature engineering & preprocessing

Hybrid model inference

Fraud metrics computation

Explainable AI visualizations

🛠️ Technology Stack
Backend

Python 3.10+

Django 5.x

SQLite (development) / MySQL (production-ready)

Machine Learning

TensorFlow / Keras

Scikit-learn

XGBoost

SHAP

LIME

Frontend

HTML5

CSS3

JavaScript

Bootstrap 5

Chart.js / Plotly

📂 Project Structure
ad_click_fraud_classification/
│
├── ad_click_fraud_classification/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── fraudapp/
│   ├── ml_engine/
│   │   ├── data_pipeline/
│   │   │   ├── data_validation.py
│   │   │   ├── preprocessing.py
│   │   │   ├── features.py
│   │   │   └── sequence_builder.py
│   │   │
│   │   ├── models/
│   │   │   ├── cnn_model.py
│   │   │   ├── rnn_model.py
│   │   │   ├── xgb_model.py
│   │   │   └── hybrid_model.py
│   │   │
│   │   ├── training/
│   │   │   ├── trainer.py
│   │   │   └── model_registry.py
│   │   │
│   │   ├── inference/
│   │   │   └── predictor.py
│   │   │
│   │   ├── explainability/
│   │   │   ├── shap_explainer.py
│   │   │   ├── lime_explainer.py
│   │   │   └── explanation_report.py
│   │   │
│   │   ├── evaluation.py
│   │   ├── pipeline.py
│   │   ├── logger.py
│   │   └── exceptions.py
│   │
│   ├── views.py
│   ├── urls.py
│   ├── models.py
│   ├── admin.py
│   └── apps.py
│
├── templates/
│   ├── login.html
│   ├── upload.html
│   ├── dashboard.html
│   └── results.html
│
├── static/
│   ├── css/style.css
│   └── js/dashboard.js
│
├── media/uploads/
├── manage.py
├── requirements.txt
└── README.md

🔄 Workflow

User logs into the system

Uploads CSV clickstream dataset

Dataset validation & preprocessing

Feature engineering & sequence construction

CNN extracts spatial patterns

RNN models temporal click behavior

XGBoost performs final classification

SHAP & LIME generate explanations

Results displayed in dashboard

📊 Key Features
✔ Fraud Detection

Binary classification (Fraud / Legitimate)

✔ Feature Engineering

Click frequency per IP/device

Time gap between clicks

Session duration

Device–IP inconsistency

Repetitive click patterns

✔ Explainable AI

SHAP: Global feature importance

LIME: Instance-level explanations

✔ Robust Engineering

Centralized logging

Custom exception handling

Modular ML pipeline

Secure file handling
📈 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

ROC-AUC

(Reported results align with research paper benchmarks)

⚙️ Installation & Setup
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Create admin user
python manage.py createsuperuser

# Start server
python manage.py runserver


Access:
👉 http://127.0.0.1:8000/

🧪 Dataset Requirements

Required CSV columns:

device_id, ip_address, click_time, app, channel, is_attributed


Supports:

TalkingData Ad Fraud Dataset

Proprietary clickstream logs

🔐 Ethics & Transparency

Explainable predictions (SHAP & LIME)

No black-box decisions

Aligns with ethical AI standards

Suitable for cybersecurity & advertising analytics


training
python -m fraudapp.ml_engine.training.run_training

to remove pycache
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"



python manage.py runserver --noreload

database setup
python manage.py makemigrations
python manage.py migrate

create admin user
python manage.py createsuperuser


pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu \
--index-url https://download.pytorch.org/whl/cpu
