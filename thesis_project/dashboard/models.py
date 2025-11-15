from django.db import models
from django.contrib.auth import get_user_model
import json

User = get_user_model()

class DashboardStatistics(models.Model):
    date = models.DateField(auto_now_add=True)
    total_predictions = models.IntegerField(default=0)
    average_likelihood = models.FloatField(default=0.0)
    at_risk_students = models.IntegerField(default=0)
    likely_to_pass = models.IntegerField(default=0)
    pass_rate = models.FloatField(default=0.0)
    
    feature_importance = models.JSONField(default=dict)
    model_performance = models.JSONField(default=dict)
    user_statistics = models.JSONField(default=dict)
    trend_data = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'dashboard_statistics'
        ordering = ['-date']
        verbose_name = 'Dashboard Statistic'
        verbose_name_plural = 'Dashboard Statistics'
    
    def __str__(self):
        return f"Dashboard Stats - {self.date}"

class MonthlyTrend(models.Model):
    year = models.IntegerField()
    month = models.IntegerField()
    total_predictions = models.IntegerField(default=0)
    average_probability = models.FloatField(default=0.0)
    pass_rate = models.FloatField(default=0.0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'monthly_trends'
        ordering = ['year', 'month']
        unique_together = ['year', 'month']
        verbose_name = 'Monthly Trend'
        verbose_name_plural = 'Monthly Trends'
    
    def __str__(self):
        return f"{self.year}-{self.month:02d}"