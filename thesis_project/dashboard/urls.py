from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    # Web pages
    path('', views.dashboard_home, name='home'),
    
    # API endpoints
    path('api/dashboard/stats/', views.dashboard_stats, name='api_stats'),
    path('api/dashboard/export-csv/', views.export_csv, name='export_csv'),
    path('api/dashboard/export-pdf/', views.export_pdf, name='export_pdf'),
]