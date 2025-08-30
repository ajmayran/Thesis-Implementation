from django.shortcuts import render
from django.http import JsonResponse

def get_dashboard_stats(request):
    """Get dashboard statistics"""
    try:
        # Mock data for dashboard - replace with actual database queries
        stats = {
            'kpi_metrics': {
                'total_predictions': 2000,
                'average_likelihood': 72.5,
                'at_risk_students': 560,
                'likely_to_pass': 1440
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
                'Study Hours': 0.52,
                'Review Center': 0.44,
                'Sleeping Hours': 0.33,
                'Anxiety Level': 0.22,
                'Age': 0.12
            },
            'trend_data': {
                'months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'likelihood': [12, 28, 34, 45, 52, 60, 63, 66, 68, 70, 72, 75]
            },
            'user_statistics': {
                'average_gpa': 2.51,
                'common_study_hours': '10 - 15',
                'average_sleeping_hours': 6.8,
                'common_review_center': 'Yes'
            }
        }
        
        return JsonResponse(stats)
        
    except Exception as e:
        return JsonResponse({
            'error': f'Dashboard stats error: {str(e)}'
        }, status=500)