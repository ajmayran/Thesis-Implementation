from django.contrib import admin
from .models import Prediction, PredictionHistory, ModelPerformance

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'prediction_result', 'probability', 'gpa', 'created_at']
    list_filter = ['prediction_result', 'review_center', 'created_at']
    search_fields = ['user__username', 'user__email', 'user__student_id']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'avg_probability', 'risk_level', 'created_at']
    list_filter = ['risk_level', 'created_at']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']

@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ['model_name', 'model_type', 'accuracy', 'precision', 'recall', 'f1_score', 'trained_at']
    list_filter = ['model_type', 'is_active', 'trained_at']
    search_fields = ['model_name']
    ordering = ['-trained_at']