from django.urls import path
from . import views

urlpatterns = [
    path('form/', views.prediction_form, name='prediction_form'),
    path('predict/', views.make_prediction, name='make_prediction'),
    path('train/', views.train_models_view, name='train_models'),
    path('stats/', views.get_prediction_stats, name='prediction_stats'),
]