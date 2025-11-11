from django.urls import path
from . import views

app_name = 'prediction'

urlpatterns = [
    # Web pages
    path('', views.predict_view, name='predict'),
    path('result/', views.result_view, name='result'),
    path('history/', views.history, name='history'),
    path('detail/<int:prediction_id>/', views.detail_view, name='detail'),
    path('delete/<int:prediction_id>/', views.delete_prediction, name='delete'),

    # API endpoints
    path('api/predict/', views.predict_exam_result, name='api_predict'),
    path('api/model-status/', views.model_status, name='model_status'),
]