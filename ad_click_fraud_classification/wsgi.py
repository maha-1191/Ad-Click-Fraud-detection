"""
WSGI config for ad_click_fraud_classification project.

It exposes the WSGI callable as a module-level variable named ``application``.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "ad_click_fraud_classification.settings"
)

application = get_wsgi_application()

