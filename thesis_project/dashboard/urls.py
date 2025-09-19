from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('stats/', views.get_dashboard_stats, name='dashboard_stats'),
]