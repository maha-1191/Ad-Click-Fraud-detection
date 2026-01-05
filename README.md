🚨 Ad Click Fraud Classification System
📌 About the Project

This project is an Ad Click Fraud Classification System designed to identify fraudulent and legitimate ad clicks in digital advertising platforms.

Online advertisements are often targeted by 🤖 bots and ⚠️ malicious users who generate fake clicks. These fake clicks increase advertising costs and reduce campaign performance.
To solve this problem, the system uses a hybrid deep learning approach that learns user click behavior and detects fraud accurately.

🧠 Hybrid Learning Approach

The system combines multiple models to improve accuracy:

🧩 CNN – Learns spatial click patterns

⏱️ RNN / LSTM – Understands time-based click behavior

🌳 XGBoost – Makes the final fraud classification decision

To make predictions transparent and trustworthy, the system uses 🔍 SHAP, which explains why a click is classified as fraud or legitimate.

🌐 Web Application Features

The entire solution is built as a Django web application, where users can:

📂 Upload clickstream CSV files

📊 Analyze fraud detection results

📈 View fraud metrics and charts

🧠 Understand model decisions using explainable AI

This project is suitable for real-world applications, not just academic experiments, and follows good software engineering practices such as modular design, logging, and error handling.

⭐ Key Highlights

✅ Detects ad click fraud accurately

🧠 Uses hybrid deep learning (CNN + RNN + XGBoost)

🔍 Provides explainable predictions using SHAP

🖥️ Web-based system with interactive dashboard

🚀 Scalable and production-ready design

ad_click_fraud_classification/
│
├── ad_click_fraud_classification/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── fraudapp/
│   ├── migrations/
│   │   └── __init__.py
│   │
│   ├── ml_engine/
│   │   ├── artifacts/
│   │   │   ├── deep_model.pt
│   │   │   ├── threshold.joblib
│   │   │   └── xgb.joblib
│   │   │
│   │   ├── data_data/
│   │   │   └── train.csv
│   │   │
│   │   ├── data_pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── data_validation.py
│   │   │   ├── preprocessing.py
│   │   │   ├── features.py
│   │   │   └── sequence_builder.py
│   │   │
│   │   ├── explainability/
│   │   │   ├── __init__.py
│   │   │   └── shap_explainer.py
│   │   │
│   │   ├── inference/
│   │   │   ├── __init__.py
│   │   │   └── predictor.py
│   │   │
│   │   ├── logs/
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── cnn_model.py
│   │   │   ├── rnn_model.py
│   │   │   ├── cnn_rnn.py
│   │   │   └── xgb_model.py
│   │   │
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── deep_trainer.py
│   │   │   ├── model_registry.py
│   │   │   ├── run_training.py
│   │   │   ├── smote_handler.py
│   │   │   └── training_config.py
│   │   │
│   │   ├── evaluation.py
│   │   ├── pipeline.py
│   │   ├── logger.py
│   │   └── exceptions.py
│   │
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── urls.py
│   ├── utils.py
│   └── views.py
│
├── templates/
│   ├── login.html
│   ├── upload.html
│   ├── dashboard.html
│   └── results.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── dashboard.js
│
├── media/
│
├── venv/
│
├── .gitignore
├── .python-version
├── build.sh
├── start.sh
├── db.sqlite3
├── manage.py
├── requirements.txt
└── README.md



⚙️ Local Setup

Follow the steps below to run the project locally.

🔹 Create virtual environment
python -m venv venv

🔹 Activate virtual environment
venv\Scripts\activate

🔹 Install required packages
pip install -r requirements.txt

🔹 Setup database
python manage.py makemigrations
python manage.py migrate

🔹 Train fraud detection model
python -m fraudapp.ml_engine.training.run_training

🔹 Start the project (no auto reload)
python manage.py runserver --noreload

🌍 Access the Application

🏠 Application: http://127.0.0.1:8000/

🔐 Admin Panel: http://127.0.0.1:8000/admin/






