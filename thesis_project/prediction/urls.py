from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_exam_result, name='predict'),
    path('model-status/', views.model_status, name='model_status'),
    path('train/', views.train_models_view, name='train_models'),
]