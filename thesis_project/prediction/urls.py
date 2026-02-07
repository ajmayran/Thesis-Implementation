from django.urls import path
from . import views

app_name = 'prediction'

urlpatterns = [
    # Student routes
    path('student/predict/', views.student_predict_view, name='student_predict'),
    path('student/history/', views.history_view, name='history'),
    path('student/result/<int:prediction_id>/', views.result_view, name='result'),
    path('student/detail/<int:prediction_id>/', views.detail_view, name='detail'),
    path('student/delete/<int:prediction_id>/', views.delete_prediction, name='delete'),
    
    # Admin routes
    path('admin/predict/', views.admin_predict_view, name='admin_predict'),
    path('admin/result/<int:prediction_id>/', views.result_view, name='admin_result'),
    path('admin/detail/<int:prediction_id>/', views.detail_view, name='admin_detail'),
    
    # Legacy route for backwards compatibility
    path('', views.predict_view, name='predict'),
    path('result/<int:prediction_id>/', views.result_view, name='legacy_result'),
    path('detail/<int:prediction_id>/', views.detail_view, name='legacy_detail'),
    path('delete/<int:prediction_id>/', views.delete_prediction, name='legacy_delete'),

    # API endpoints
    path('api/predict/', views.predict_exam_result, name='api_predict'),
    path('api/model-status/', views.model_status, name='model_status'),
]