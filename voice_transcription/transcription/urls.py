# transcription/urls.py
from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/inference/", views.api_inference, name="api_inference"),
]
