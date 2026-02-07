from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.dashboard_home, name='home'),
    path('student/', views.student_dashboard, name='student_dashboard'),
    path('student/profile/', views.student_profile, name='student_profile'),
    path('reports/', views.reports_view, name='reports'),
    
    # API endpoints for dashboard
    path('api/dashboard/stats/', views.dashboard_stats, name='dashboard_stats'),
    path('api/dashboard/trends/', views.get_trend_data, name='trend_data'),
    path('api/dashboard/export-csv/', views.export_csv, name='export_csv'),
    path('api/dashboard/export-pdf/', views.export_pdf, name='export_pdf'),
    
    # API endpoints for reports
    path('api/reports/data/', views.get_reports_data, name='reports_data'),
    path('api/reports/export-pdf/', views.export_reports_pdf, name='reports_export_pdf'),
    path('api/reports/export-csv/', views.export_reports_csv, name='reports_export_csv'),
]