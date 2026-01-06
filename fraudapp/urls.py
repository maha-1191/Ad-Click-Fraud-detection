from django.urls import path
from . import views

urlpatterns = [

    # Landing
    path("", views.home, name="home"),

    # Auth
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),

    # UI
    path("dashboard/", views.dashboard, name="dashboard"),
    path("upload/", views.upload_dataset, name="upload"),
    path("profile/", views.profile_view, name="profile"),

    # ML
    path(
        "run-detection/<int:dataset_id>/",
        views.run_detection,
        name="run_detection"
    ),

    path(
        "export/ip-blacklist/<int:dataset_id>/",
        views.export_ip_blacklist,
        name="export_ip_blacklist"
    ),

    # API
    path(
        "api/predict/",
        views.api_predict_fraud,
        name="api_predict"
    ),
]












