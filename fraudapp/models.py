from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class UploadedDataset(models.Model):
    """
    Stores uploaded clickstream datasets
    """

    STATUS_CHOICES = [
        ("UPLOADED", "Uploaded"),
        ("PROCESSED", "Processed"),
        ("FAILED", "Failed"),
    ]

    original_filename = models.CharField(max_length=255)
    file = models.FileField(upload_to="uploads/")

    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="uploaded_datasets"
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="UPLOADED"
    )

    total_rows = models.PositiveIntegerField(default=0)
    total_columns = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Dataset {self.id} - {self.original_filename}"


class PredictionResult(models.Model):
    """
    Stores inference + business intelligence output
    """

    dataset = models.OneToOneField(
        UploadedDataset,
        on_delete=models.CASCADE,
        related_name="prediction_result"
    )

    total_clicks = models.PositiveIntegerField(default=0)
    fraud_clicks = models.PositiveIntegerField(default=0)
    legit_clicks = models.PositiveIntegerField(default=0)

    # Core inference summary
    metrics = models.JSONField(default=dict)

    # Advertiser intelligence
    ip_risk = models.JSONField(default=list)
    asn_risk = models.JSONField(default=list)

    # Business impact estimation
    business_impact = models.JSONField(default=dict)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"PredictionResult for Dataset {self.dataset.id}"










