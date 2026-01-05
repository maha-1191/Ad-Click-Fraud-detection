About the Project

This project is an Ad Click Fraud Classification System designed to identify fraudulent and legitimate ad clicks in digital advertising platforms.

Online advertisements are often targeted by bots or malicious users who generate fake clicks. These fake clicks increase advertising costs and reduce campaign performance. To solve this problem, this system uses a hybrid deep learning approach that learns user click behavior and detects fraud accurately.

The system combines:

CNN to learn spatial click patterns

RNN/LSTM to understand time-based click behavior

XGBoost to make the final fraud decision

To make predictions transparent and trustworthy, the system uses SHAP to explain why a click is classified as fraud or legitimate.

The entire solution is built as a Django web application where users can:

Upload clickstream CSV files

Analyze fraud results

View fraud metrics and charts

Understand model decisions using explainable AI

This project is suitable for real-world use, not just academic experiments, and follows good software engineering practices such as modular design, logging, and error handling.

Key Highlights

Detects ad click fraud accurately

Uses hybrid deep learning (CNN + RNN + XGBoost)

Provides explainable predictions using SHAP

Web-based system with interactive dashboard

Scalable and production-ready design


local setup

REM ===== Create virtual environment =====
python -m venv venv

REM ===== Activate virtual environment =====
call venv\Scripts\activate

REM ===== Install required packages =====
pip install -r requirements.txt

REM ===== Setup database =====
python manage.py makemigrations
python manage.py migrate

REM ===== Create admin user =====
python manage.py createsuperuser

REM ===== Train fraud detection model =====
python -m fraudapp.ml_engine.training.run_training

REM ===== Start the project (no auto reload) =====
python manage.py runserver --noreload







