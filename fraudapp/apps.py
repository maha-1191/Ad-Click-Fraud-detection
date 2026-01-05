from django.apps import AppConfig


class FraudappConfig(AppConfig):
    """
    Configuration for the Fraud Detection application.

    This app is responsible for:
    - Dataset uploads
    - Triggering ML pipeline execution
    - Storing predictions and metrics
    - Serving dashboard views
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "fraudapp"
    verbose_name = "Ad Click Fraud Detection"

