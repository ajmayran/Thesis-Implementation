from django.db import models
from django.utils import timezone
import json

class PredictionHistory(models.Model):
    """Store prediction history for analytics"""
    
    # Input Data
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
    
    # Prediction Results
    avg_probability = models.FloatField()
    risk_level = models.CharField(max_length=20)
    base_predictions = models.JSONField()  # Store as JSON
    ensemble_predictions = models.JSONField(null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)
    
    class Meta:
        db_table = 'prediction_history'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['risk_level']),
        ]
    
    def __str__(self):
        return f"Prediction {self.id} - {self.avg_probability:.1f}% ({self.created_at})"
    
    @property
    def prediction_outcome(self):
        """Get predicted outcome"""
        return "Pass" if self.avg_probability >= 50 else "Fail"
    
    @property
    def is_high_risk(self):
        """Check if high risk"""
        return self.avg_probability < 50

class ModelPerformance(models.Model):
    """Track model performance metrics"""
    
    model_name = models.CharField(max_length=50)
    model_type = models.CharField(max_length=20, choices=[
        ('base', 'Base Model'),
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
        unique_together = ['model_name', 'trained_at']
    
    def __str__(self):
        return f"{self.model_name} - Accuracy: {self.accuracy:.2%}"