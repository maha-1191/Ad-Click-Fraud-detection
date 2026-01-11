from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),

    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),

    path("dashboard/", views.dashboard, name="dashboard"),
    path("profile/", views.profile_view, name="profile"),

    path("upload/", views.upload_dataset, name="upload"),

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


]













