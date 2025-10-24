from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import csv
from datetime import datetime

def dashboard_home(request):
    """Render the main dashboard page"""
    return render(request, 'pages/dashboard.html')

@require_http_methods(["GET"])
def dashboard_stats(request):
    """API endpoint for dashboard statistics"""
    try:
        # Replace with real data
        stats = {
            'kpi_metrics': {
                'total_predictions': 2000,
                'average_likelihood': 72.5,
                'at_risk_students': 560,
                'likely_to_pass': 1440,
                'trends': {
                    'predictions': 12,
                    'likelihood': 3.2,
                    'at_risk': -5,
                    'likely_pass': 8
                }
            },
            'model_performance': {
                'accuracy': 83,
                'precision': 0.78,
                'recall': 0.76,
                'f1_score': 0.77,
                'auc': 0.88
            },
            'feature_importance': {
                'GPA': 0.62,
                'StudyHours': 0.52,
                'ReviewCenter': 0.44,
                'SleepingHours': 0.33,
                'TestAnxiety': 0.22,
                'Age': 0.12
            },
            'user_statistics': {
                'average_gpa': 2.51,
                'common_study_hours': '10-15',
                'average_sleep_hours': 6.8,
                'review_center_rate': 75
            },
            'trend_data': {
                'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'values': [12, 28, 34, 45, 52, 60, 63, 66, 68, 70, 72, 75]
            }
        }
        
        return JsonResponse(stats)
        
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'message': 'Failed to load dashboard statistics'
        }, status=500)

@require_http_methods(["GET"])
def export_csv(request):
    """Export predictions data to CSV"""
    try:
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="predictions_{datetime.now().strftime("%Y%m%d")}.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Student ID', 'Prediction', 'Probability', 'Risk Level', 'Date'])
        
        # Replace with actual database queries
        sample_data = [
            ['001', 'Pass', '85.2%', 'Low', '2024-01-15'],
            ['002', 'Fail', '42.1%', 'High', '2024-01-15'],
            ['003', 'Pass', '78.5%', 'Low', '2024-01-16'],
        ]
        
        for row in sample_data:
            writer.writerow(row)
        
        return response
        
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'message': 'Failed to export CSV'
        }, status=500)

@require_http_methods(["GET"])
def export_pdf(request):
    """Generate PDF report"""
    try:
        # TODO: Implement PDF generation using reportlab or similar
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="report_{datetime.now().strftime("%Y%m%d")}.pdf"'
        
        # Placeholder - implement actual PDF generation
        response.write(b'PDF generation not yet implemented')
        
        return response
        
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'message': 'Failed to generate PDF'
        }, status=500)