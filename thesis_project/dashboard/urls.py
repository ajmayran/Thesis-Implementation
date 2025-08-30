from django.urls import path
from . import views

urlpatterns = [
    path('stats/', views.get_dashboard_stats, name='dashboard_stats'),
]