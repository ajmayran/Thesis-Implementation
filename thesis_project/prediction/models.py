from django.db import models
from django.utils import timezone
from django.conf import settings

class Prediction(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='predictions')
    
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    gpa = models.FloatField()
    internship_grade = models.FloatField()
    study_hours = models.IntegerField()
    sleep_hours = models.IntegerField()
    review_center = models.BooleanField(default=False)
    confidence = models.IntegerField()
    mock_exam_score = models.FloatField(null=True, blank=True)
    scholarship = models.BooleanField(default=False)
    income_level = models.CharField(max_length=20)
    employment_status = models.CharField(max_length=20)
    
    prediction_result = models.CharField(max_length=10)
    probability = models.FloatField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'predictions'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.prediction_result} ({self.probability:.2f}%)"

class PredictionHistory(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='prediction_histories')
    
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    study_hours = models.IntegerField()
    sleep_hours = models.IntegerField()
    review_center = models.BooleanField()
    confidence = models.IntegerField()
    mock_exam_score = models.FloatField(null=True, blank=True)
    gpa = models.FloatField()
    scholarship = models.BooleanField()
    internship_grade = models.FloatField()
    income_level = models.CharField(max_length=20)
    employment_status = models.CharField(max_length=20)
    
    avg_probability = models.FloatField()
    risk_level = models.CharField(max_length=20)
    base_predictions = models.JSONField()
    ensemble_predictions = models.JSONField(null=True, blank=True)
    
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)
    
    class Meta:
        db_table = 'prediction_history'
        ordering = ['-created_at']
        verbose_name = 'Prediction History'
        verbose_name_plural = 'Prediction Histories'
    
    def __str__(self):
        return f"{self.user.username} - {self.risk_level} ({self.created_at})"

class ModelPerformance(models.Model):
    model_name = models.CharField(max_length=50)
    model_type = models.CharField(max_length=20, choices=[
        ('classification', 'Classification Model'),
        ('ensemble', 'Ensemble Model')
    ])
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    auc_score = models.FloatField(null=True, blank=True)
    cv_mean = models.FloatField(null=True, blank=True)
    cv_std = models.FloatField(null=True, blank=True)
    
    trained_at = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'model_performance'
        ordering = ['-trained_at']
        unique_together = [['model_name', 'trained_at']]
    
    def __str__(self):
        return f"{self.model_name} ({self.model_type}) - Accuracy: {self.accuracy:.2f}%"

class RegressionModelPerformance(models.Model):
    model_name = models.CharField(max_length=50)
    model_type = models.CharField(max_length=20, choices=[
        ('regression', 'Regression Model'),
        ('ensemble_regression', 'Ensemble Regression')
    ])
    rmse = models.FloatField()
    mae = models.FloatField()
    r2_score = models.FloatField()
    mse = models.FloatField()
    cv_rmse = models.FloatField(null=True, blank=True)
    cv_std = models.FloatField(null=True, blank=True)
    
    trained_at = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'regression_model_performance'
        ordering = ['-trained_at']
        unique_together = [['model_name', 'trained_at']]
    
    def __str__(self):
        return f"{self.model_name} ({self.model_type}) - RMSE: {self.rmse:.4f}"