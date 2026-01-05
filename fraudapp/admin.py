from django.contrib import admin
from .models import UploadedDataset, PredictionResult


@admin.register(UploadedDataset)
class UploadedDatasetAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "original_filename",
        "uploaded_by",
        "status",
        "total_rows",
        "created_at",
    )
    list_filter = ("status", "created_at")
    search_fields = ("original_filename",)
    ordering = ("-created_at",)


@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "dataset",
        "total_clicks",
        "fraud_clicks",
        "legit_clicks",
        "created_at",
    )

    list_filter = ("created_at",)
    ordering = ("-created_at",)

    readonly_fields = (
        "metrics",
        "ip_risk",
        "asn_risk",
        "created_at",
    )



