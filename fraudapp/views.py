import pandas as pd
import csv

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User

from fraudapp.ml_engine.inference.predictor import FraudPredictor
from fraudapp.ml_engine.logger import get_logger
from .models import UploadedDataset, PredictionResult

logger = get_logger(__name__)
_PREDICTOR = None

def get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        logger.debug("Using UPDATED FraudPredictor")
        _PREDICTOR = FraudPredictor()
    return _PREDICTOR
def home(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return render(request, "home.html")
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
@login_required
def dashboard(request):
    dataset_id = request.GET.get("dataset_id")

    dataset = None
    result = None

    if dataset_id:
        dataset = get_object_or_404(
            UploadedDataset,
            id=dataset_id,
            uploaded_by=request.user,
            status="PROCESSED"
        )
    else:
        dataset = (
            UploadedDataset.objects
            .filter(uploaded_by=request.user, status="PROCESSED")
            .order_by("-created_at")
            .first()
        )

    if dataset:
        result = PredictionResult.objects.filter(
            dataset=dataset
        ).first()

    return render(
        request,
        "dashboard.html",
        {
            "dataset": dataset,
            "result": result,
        }
    )

   
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
@login_required
def upload_dataset(request):
    if request.method == "POST":
        try:
            file = request.FILES.get("dataset")

            if not file or not file.name.endswith(".csv"):
                messages.error(request, "Please upload a CSV file.")
                return render(request, "upload.html")

            dataset = UploadedDataset.objects.create(
                original_filename=file.name,
                file=file,
                uploaded_by=request.user,
                status="UPLOADED",
            )

            df = pd.read_csv(dataset.file.path)
            if df.empty:
                dataset.status = "FAILED"
                dataset.save()
                messages.error(request, "CSV is empty.")
                return render(request, "upload.html")

            dataset.total_rows = len(df)
            dataset.total_columns = len(df.columns)
            dataset.save()

            messages.success(request, "Dataset uploaded successfully.")
            return render(request, "upload.html", {"dataset": dataset})

        except Exception:
            logger.exception("Upload failed")
            messages.error(request, "Upload failed.")
            return render(request, "upload.html")

    return render(request, "upload.html")


@login_required
def run_detection(request, dataset_id):
    dataset = get_object_or_404(
        UploadedDataset,
        id=dataset_id,
        uploaded_by=request.user
    )

    try:
        df = pd.read_csv(dataset.file.path)
        predictor = get_predictor()
        output = predictor.predict(df)

        summary = output["summary"]

        PredictionResult.objects.create(
            dataset=dataset,

            total_clicks=summary["total_clicks"],
            fraud_clicks=summary["fraud_clicks"],
            legit_clicks=summary["legit_clicks"],

            metrics={
                "summary": summary,
                "business_impact": output.get("business_impact", {}),
                "time_trends": output.get("time_trends", []),
                "shap_summary": output.get("shap_summary", []),
            },

            ip_risk=output.get("ip_risk", []),
        )

        dataset.status = "PROCESSED"
        dataset.save()

        messages.success(request, "Fraud detection completed.")
        return redirect("dashboard")

    except Exception:
        logger.exception("Fraud detection failed")
        dataset.status = "FAILED"
        dataset.save()
        messages.error(request, "Fraud detection failed.")
        return redirect("dashboard")


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

@csrf_exempt
@require_http_methods(["POST"])
def api_predict_fraud(request):
    try:
        df = pd.read_csv(request.FILES["file"])
        predictor = get_predictor()
        output = predictor.predict(df)
        return JsonResponse({"status": "success", "data": output})
    except Exception as e:
        logger.exception("API inference failed")
        return JsonResponse(
            {"status": "error", "message": str(e)},
            status=500
        )









