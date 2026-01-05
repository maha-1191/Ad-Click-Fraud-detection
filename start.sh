#!/usr/bin/env bash
python manage.py migrate
python manage.py collectstatic --noinput
gunicorn ad_click_fraud_classification.wsgi:application
