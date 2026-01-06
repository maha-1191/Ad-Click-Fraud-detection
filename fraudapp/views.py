import csv
import pandas as pd

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.http import HttpResponse
from django.contrib.auth.models import User

from .models import UploadedDataset, PredictionResult


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
                "Dataset uploaded. Open ML service to run detection."
            )

            return render(request, "upload.html", {"dataset": dataset})

        except Exception:
            messages.error(request, "Upload failed.")
            return render(request, "upload.html")

    return render(request, "upload.html")


# =====================================================
# RUN FRAUD DETECTION (NO API – REDIRECT ONLY)
# =====================================================
@login_required
def run_detection(request, dataset_id):
    dataset = get_object_or_404(
        UploadedDataset,
        id=dataset_id,
        uploaded_by=request.user
    )

    dataset.status = "PROCESSED"
    dataset.save()

    messages.info(
        request,
        "Fraud detection runs in the ML service. "
        "Please upload the dataset there."
    )

    return redirect(
        "https://ad-click-fraud-detection-8df3vwi47neaz53utto84g.streamlit.app"
    )


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






















