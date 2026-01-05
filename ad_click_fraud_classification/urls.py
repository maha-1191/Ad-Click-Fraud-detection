from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),

    # Delegate ALL routing to fraudapp
    path("", include("fraudapp.urls")),
]




