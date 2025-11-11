from django.contrib import admin
from .models import DashboardStatistics, MonthlyTrend

@admin.register(DashboardStatistics)
class DashboardStatisticsAdmin(admin.ModelAdmin):
    list_display = ['date', 'total_predictions', 'average_likelihood', 'at_risk_students', 'likely_to_pass', 'pass_rate']
    list_filter = ['date']
    search_fields = ['date']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-date']

@admin.register(MonthlyTrend)
class MonthlyTrendAdmin(admin.ModelAdmin):
    list_display = ['year', 'month', 'total_predictions', 'average_probability', 'pass_rate']
    list_filter = ['year', 'month']
    ordering = ['-year', '-month']