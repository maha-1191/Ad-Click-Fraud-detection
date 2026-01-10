import csv
import pandas as pd
import os
import requests

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.models import User

from .models import UploadedDataset, PredictionResult
from .ml_engine.inference.predictor import FraudPredictor


# =====================================================
# HOME
# =====================================================
def home(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return render(request, "home.html")


# =====================================================
# AUTH
# =====================================================
@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        user = authenticate(
            request,
            username=request.POST.get("username"),
            password=request.POST.get("password"),
        )
        if user:
            login(request, user)
            return redirect("dashboard")

        messages.error(request, "Invalid credentials")

    return render(request, "login.html")


@require_http_methods(["GET", "POST"])
def register_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        if request.POST["password1"] != request.POST["password2"]:
            messages.error(request, "Passwords do not match")
            return render(request, "register.html")

        User.objects.create_user(
            username=request.POST["username"],
            email=request.POST.get("email"),
            password=request.POST["password1"],
        )

        messages.success(request, "Account created. Please login.")
        return redirect("login")

    return render(request, "register.html")


@login_required
def logout_view(request):
    logout(request)
    return redirect("login")


# =====================================================
# DASHBOARD
# =====================================================
@login_required
def dashboard(request):
    dataset = (
        UploadedDataset.objects
        .filter(uploaded_by=request.user)
        .order_by("-created_at")
        .first()
    )

    result = None
    if dataset:
        result = PredictionResult.objects.filter(dataset=dataset).first()

    return render(
        request,
        "dashboard.html",
        {
            "dataset": dataset,
            "result": result,
        }
    )


# =====================================================
# UPLOAD DATASET
# =====================================================
@login_required
def upload_dataset(request):
    if request.method == "POST":
        try:
            file = request.FILES.get("dataset")

            if not file or not file.name.endswith(".csv"):
                messages.error(request, "Upload a valid CSV file.")
                return render(request, "upload.html")

            dataset = UploadedDataset.objects.create(
                original_filename=file.name,
                file=file,
                uploaded_by=request.user,
                status="UPLOADED",
            )

            df = pd.read_csv(dataset.file.path)
            dataset.total_rows = len(df)
            dataset.total_columns = len(df.columns)
            dataset.save()

            messages.success(
                request,
                "Dataset uploaded successfully. Click 'Run Detection'."
            )

            return render(request, "upload.html", {"dataset": dataset})

        except Exception as e:
            messages.error(request, f"Upload failed: {e}")
            return render(request, "upload.html")

    return render(request, "upload.html")


# =====================================================
# RUN FRAUD DETECTION (FASTAPI ON RENDER)
# =====================================================
@login_required
def run_detection(request, dataset_id):
    dataset = get_object_or_404(
        UploadedDataset,
        id=dataset_id,
        uploaded_by=request.user
    )

    ML_API_URL = os.getenv(
        "ML_API_URL",
        "https://fraud-ml-api.onrender.com/predict"
    ).strip()

    try:
        with open(dataset.file.path, "rb") as f:
            response = requests.post(
                ML_API_URL,
                files={
                    "file": (
                        dataset.original_filename,
                        f,
                        "text/csv"
                    )
                },
                timeout=300
            )

        response.raise_for_status()
        data = response.json()

        summary = data.get("summary", {})

        PredictionResult.objects.create(
            dataset=dataset,

            total_clicks=summary.get("total_clicks", 0),
            fraud_clicks=summary.get("fraud_clicks", 0),
            legit_clicks=summary.get("legit_clicks", 0),

            metrics=summary,
            ip_risk=data.get("ip_risk", []),
            business_impact=data.get("business_impact", {}),
            time_trends=data.get("time_trends", []),
            shap_summary=data.get("shap_summary", []),
        )

        dataset.status = "PROCESSED"
        dataset.save()

        messages.success(request, "Fraud detection completed successfully.")

    except Exception as e:
        dataset.status = "FAILED"
        dataset.save()
        messages.error(request, f"Fraud detection failed: {e}")

    return redirect("dashboard")


# =====================================================
# EXPORT IP BLACKLIST
# =====================================================
@login_required
def export_ip_blacklist(request, dataset_id):
    result = get_object_or_404(PredictionResult, dataset_id=dataset_id)

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = (
        f'attachment; filename="ip_blacklist_{dataset_id}.csv"'
    )

    writer = csv.writer(response)
    writer.writerow([
        "ip",
        "avg_risk_score",
        "risk_level",
        "recommended_action"
    ])

    for ip in result.ip_risk:
        writer.writerow([
            ip.get("ip"),
            ip.get("avg_risk_score"),
            ip.get("risk_level"),
            ip.get("recommended_action"),
        ])

    return response


# =====================================================
# PROFILE
# =====================================================
@login_required
def profile_view(request):
    datasets = (
        UploadedDataset.objects
        .filter(uploaded_by=request.user)
        .order_by("-created_at")
    )

    return render(
        request,
        "profile.html",
        {
            "user_obj": request.user,
            "datasets": datasets,
            "total_datasets": datasets.count(),
        }
    )


# =====================================================
# LOCAL PREDICT API (OPTIONAL)
# =====================================================
@csrf_exempt
def predict_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    try:
        file = request.FILES["file"]
        df = pd.read_csv(file)

        predictor = FraudPredictor()
        result = predictor.predict(df)

        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


