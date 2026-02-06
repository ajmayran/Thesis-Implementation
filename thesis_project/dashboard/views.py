from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg, Count, Q, Max, Min
from django.db.models.functions import TruncMonth, ExtractMonth, ExtractYear
from prediction.models import Prediction, RegressionModelPerformance
from .models import DashboardStatistics, MonthlyTrend
from django.utils import timezone
import json
import csv
from datetime import datetime, timedelta
from collections import Counter

@login_required
def dashboard_home(request):
    if request.user.role == 'student':
        return redirect('dashboard:student_dashboard')
    
    return render(request, 'pages/dashboard.html')

@login_required
def student_dashboard(request):
    if request.user.role == 'admin' or request.user.is_superuser:
        return redirect('dashboard:home')
    
    predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
    
    total_predictions = predictions.count()
    latest_prediction = predictions.first()
    recent_predictions = predictions[:5]
    
    avg_prob = predictions.aggregate(Avg('probability'))['probability__avg'] or 0
    highest_prob = predictions.order_by('-probability').first()
    highest_probability = highest_prob.probability if highest_prob else 0
    
    chart_predictions = predictions[:10][::-1]
    chart_labels = [p.created_at.strftime('%m/%d') for p in chart_predictions]
    chart_data = [float(p.probability) for p in chart_predictions]
    
    context = {
        'total_predictions': total_predictions,
        'latest_prediction': latest_prediction,
        'recent_predictions': recent_predictions,
        'average_probability': avg_prob,
        'highest_probability': highest_probability,
        'chart_labels': json.dumps(chart_labels),
        'chart_data': json.dumps(chart_data),
    }
    
    return render(request, 'pages/student_dashboard.html', context)

@login_required
def student_profile(request):
    return render(request, 'pages/student_profile.html')

def load_feature_importance_from_json():
    """Load feature importance from the analysis JSON file"""
    import os
    
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'models',
        'regression_processed_data',
        'feature_importance_analysis.json'
    )
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Use combined ranking average rank (lower is better)
        combined = data.get('combined_ranking', [])
        
        # Convert to importance scores (inverse of rank, normalized)
        feature_importance = {}
        if combined:
            max_rank = max(item['Avg_Rank'] for item in combined)
            for item in combined:
                # Inverse ranking: lower rank = higher importance
                importance = (max_rank - item['Avg_Rank'] + 1) / max_rank
                feature_importance[item['Feature']] = round(importance, 4)
        
        return feature_importance
    except Exception as e:
        print(f"Error loading feature importance: {e}")
        # Fallback to default values
        return {
            'Age': 0.85,
            'TestAnxiety': 0.72,
            'SleepHours': 0.68,
            'StudyHours': 0.58,
            'GPA': 0.52,
            'EnglishProficiency': 0.48
        }

def calculate_dashboard_statistics():
    all_predictions = Prediction.objects.all()
    
    total_predictions = all_predictions.count()
    
    if total_predictions == 0:
        return None
    
    avg_probability = all_predictions.aggregate(Avg('probability'))['probability__avg'] or 0
    
    # At-risk: probability < 50%
    at_risk_students = all_predictions.filter(probability__lt=50).values('user').distinct().count()
    # Likely to pass: probability >= 75%
    likely_to_pass = all_predictions.filter(probability__gte=75).values('user').distinct().count()
    
    pass_rate = (all_predictions.filter(probability__gte=50).count() / total_predictions * 100) if total_predictions > 0 else 0
    
    # Load feature importance from JSON file
    feature_importance = load_feature_importance_from_json()
    
    # Get latest regression model performance
    latest_model = RegressionModelPerformance.objects.filter(
        is_active=True
    ).order_by('-trained_at').first()
    
    if latest_model:
        model_performance = {
            'rmse': round(latest_model.rmse, 4),
            'mae': round(latest_model.mae, 4),
            'r2_score': round(latest_model.r2_score, 4),
            'mse': round(latest_model.mse, 4),
            'cv_rmse': round(latest_model.cv_rmse, 4) if latest_model.cv_rmse else None,
            'cv_std': round(latest_model.cv_std, 4) if latest_model.cv_std else None,
            'model_name': latest_model.model_name,
            'model_type': latest_model.model_type,
            'trained_at': latest_model.trained_at.strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        # No model available
        model_performance = {
            'rmse': 0,
            'mae': 0,
            'r2_score': 0,
            'mse': 0,
            'cv_rmse': None,
            'cv_std': None,
            'model_name': 'No Model',
            'model_type': 'N/A',
            'trained_at': 'Not trained yet'
        }
    
    # Calculate user statistics
    user_stats = all_predictions.aggregate(
        avg_gpa=Avg('gpa'),
        avg_study=Avg('study_hours'),
        avg_sleep=Avg('sleep_hours'),
        avg_confidence=Avg('confidence'),
        avg_test_anxiety=Avg('test_anxiety'),
        avg_mock_score=Avg('mock_exam_score'),
        avg_internship=Avg('internship_grade'),
        avg_age=Avg('age')
    )
    
    # Review center statistics
    review_center_count = all_predictions.filter(review_center=True).count()
    review_center_rate = (review_center_count / total_predictions * 100) if total_predictions > 0 else 0
    
    # Scholarship statistics
    scholarship_count = all_predictions.filter(scholarship=True).count()
    scholarship_rate = (scholarship_count / total_predictions * 100) if total_predictions > 0 else 0
    
    # Most common study hours
    study_hours_list = all_predictions.values_list('study_hours', flat=True)
    study_hours_counter = Counter(study_hours_list)
    most_common_study = study_hours_counter.most_common(1)[0][0] if study_hours_counter else 0
    
    # Gender distribution
    gender_counts = all_predictions.values('gender').annotate(count=Count('id'))
    gender_distribution = {item['gender']: item['count'] for item in gender_counts}
    
    # Employment status distribution
    employment_counts = all_predictions.values('employment_status').annotate(count=Count('id'))
    employment_distribution = {item['employment_status']: item['count'] for item in employment_counts}
    
    user_statistics = {
        'average_gpa': round(user_stats['avg_gpa'] or 0, 2),
        'average_internship_grade': round(user_stats['avg_internship'] or 0, 2),
        'common_study_hours': f"{most_common_study} hours",
        'average_study_hours': round(user_stats['avg_study'] or 0, 1),
        'average_sleep_hours': round(user_stats['avg_sleep'] or 0, 1),
        'average_age': round(user_stats['avg_age'] or 0, 1),
        'review_center_rate': round(review_center_rate, 1),
        'scholarship_rate': round(scholarship_rate, 1),
        'average_confidence': round(user_stats['avg_confidence'] or 0, 1),
        'average_test_anxiety': round(user_stats['avg_test_anxiety'] or 0, 1),
        'average_mock_score': round(user_stats['avg_mock_score'] or 0, 1),
        'gender_distribution': gender_distribution,
        'employment_distribution': employment_distribution,
        'total_students': all_predictions.values('user').distinct().count()
    }
    
    # Calculate trends
    thirty_days_ago = timezone.now() - timedelta(days=30)
    previous_total = all_predictions.filter(created_at__lt=thirty_days_ago).count()
    current_total = all_predictions.filter(created_at__gte=thirty_days_ago).count()
    
    trend_predictions = ((current_total - previous_total) / previous_total * 100) if previous_total > 0 else 0
    
    previous_avg = all_predictions.filter(created_at__lt=thirty_days_ago).aggregate(Avg('probability'))['probability__avg'] or 0
    current_avg = all_predictions.filter(created_at__gte=thirty_days_ago).aggregate(Avg('probability'))['probability__avg'] or 0
    trend_likelihood = current_avg - previous_avg
    
    previous_at_risk = all_predictions.filter(created_at__lt=thirty_days_ago, probability__lt=50).count()
    current_at_risk = all_predictions.filter(created_at__gte=thirty_days_ago, probability__lt=50).count()
    trend_at_risk = ((current_at_risk - previous_at_risk) / previous_at_risk * 100) if previous_at_risk > 0 else 0
    
    previous_likely = all_predictions.filter(created_at__lt=thirty_days_ago, probability__gte=75).count()
    current_likely = all_predictions.filter(created_at__gte=thirty_days_ago, probability__gte=75).count()
    trend_likely = ((current_likely - previous_likely) / previous_likely * 100) if previous_likely > 0 else 0
    
    return {
        'kpi_metrics': {
            'total_predictions': total_predictions,
            'average_likelihood': round(avg_probability, 1),
            'at_risk_students': at_risk_students,
            'likely_to_pass': likely_to_pass,
            'pass_rate': round(pass_rate, 1),
            'trends': {
                'predictions': round(trend_predictions, 1),
                'likelihood': round(trend_likelihood, 1),
                'at_risk': round(trend_at_risk, 1),
                'likely_pass': round(trend_likely, 1)
            }
        },
        'model_performance': model_performance,
        'feature_importance': feature_importance,
        'user_statistics': user_statistics
    }

def calculate_monthly_trends():
    now = timezone.now()
    twelve_months_ago = now - timedelta(days=365)
    
    monthly_data = Prediction.objects.filter(
        created_at__gte=twelve_months_ago
    ).annotate(
        month=ExtractMonth('created_at'),
        year=ExtractYear('created_at')
    ).values('year', 'month').annotate(
        count=Count('id'),
        avg_prob=Avg('probability')
    ).order_by('year', 'month')
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    labels = []
    values = []
    
    for data in monthly_data:
        month_name = months[data['month'] - 1]
        labels.append(month_name)
        values.append(round(data['avg_prob'], 1))
    
    # Fill missing months with 0
    if len(labels) < 12:
        for i in range(12 - len(labels)):
            month_idx = (now.month - 12 + i) % 12
            labels.insert(0, months[month_idx])
            values.insert(0, 0)
    
    return {
        'labels': labels[-12:],
        'values': values[-12:]
    }

@require_http_methods(["GET"])
@login_required
def dashboard_stats(request):
    try:
        stats = calculate_dashboard_statistics()
        
        if not stats:
            return JsonResponse({
                'kpi_metrics': {
                    'total_predictions': 0,
                    'average_likelihood': 0,
                    'at_risk_students': 0,
                    'likely_to_pass': 0,
                    'pass_rate': 0,
                    'trends': {
                        'predictions': 0,
                        'likelihood': 0,
                        'at_risk': 0,
                        'likely_pass': 0
                    }
                },
                'model_performance': {
                    'rmse': 0,
                    'mae': 0,
                    'r2_score': 0,
                    'mse': 0,
                    'cv_rmse': None,
                    'cv_std': None,
                    'model_name': 'No Model',
                    'model_type': 'N/A',
                    'trained_at': 'Not trained yet'
                },
                'feature_importance': {},
                'user_statistics': {
                    'average_gpa': 0,
                    'average_internship_grade': 0,
                    'common_study_hours': '0 hours',
                    'average_study_hours': 0,
                    'average_sleep_hours': 0,
                    'average_age': 0,
                    'review_center_rate': 0,
                    'scholarship_rate': 0,
                    'average_confidence': 0,
                    'average_test_anxiety': 0,
                    'average_mock_score': 0,
                    'total_students': 0
                },
                'trend_data': {
                    'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    'values': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                }
            })
        
        trend_data = calculate_monthly_trends()
        stats['trend_data'] = trend_data
        
        # Save to dashboard statistics
        today = timezone.now().date()
        DashboardStatistics.objects.update_or_create(
            date=today,
            defaults={
                'total_predictions': stats['kpi_metrics']['total_predictions'],
                'average_likelihood': stats['kpi_metrics']['average_likelihood'],
                'at_risk_students': stats['kpi_metrics']['at_risk_students'],
                'likely_to_pass': stats['kpi_metrics']['likely_to_pass'],
                'pass_rate': stats['kpi_metrics']['pass_rate'],
                'feature_importance': stats['feature_importance'],
                'model_performance': stats['model_performance'],
                'user_statistics': stats['user_statistics'],
                'trend_data': trend_data
            }
        )
        
        return JsonResponse(stats)
        
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'message': 'Failed to load dashboard statistics'
        }, status=500)

@require_http_methods(["GET"])
@login_required
def export_csv(request):
    try:
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
        
        writer = csv.writer(response)
        writer.writerow([
            'Student ID', 'Student Name', 'Email', 'Prediction Result', 
            'Probability (%)', 'Age', 'Gender', 'GPA', 'Internship Grade',
            'Study Hours', 'Sleep Hours', 'Review Center', 'Confidence', 
            'Test Anxiety', 'Mock Exam Score', 'Scholarship', 'Income Level',
            'Employment Status', 'Date'
        ])
        
        predictions = Prediction.objects.select_related('user').order_by('-created_at')
        
        for pred in predictions:
            writer.writerow([
                pred.user.student_id or 'N/A',
                pred.user.get_full_name(),
                pred.user.email,
                pred.prediction_result,
                f'{pred.probability:.1f}',
                pred.age,
                pred.gender,
                f'{pred.gpa:.2f}',
                f'{pred.internship_grade:.2f}',
                pred.study_hours,
                pred.sleep_hours,
                'Yes' if pred.review_center else 'No',
                pred.confidence,
                pred.test_anxiety,
                f'{pred.mock_exam_score:.1f}' if pred.mock_exam_score else 'N/A',
                'Yes' if pred.scholarship else 'No',
                pred.income_level,
                pred.employment_status,
                pred.created_at.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        return response
        
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'message': 'Failed to export CSV'
        }, status=500)

@require_http_methods(["GET"])
@login_required
def export_pdf(request):
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from io import BytesIO
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        elements = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f2937'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#374151'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        elements.append(Paragraph('Social Work Licensure Exam Predictor', title_style))
        elements.append(Paragraph('Dashboard Report', styles['Heading2']))
        elements.append(Paragraph(f'Generated: {datetime.now().strftime("%B %d, %Y %H:%M:%S")}', styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        stats = calculate_dashboard_statistics()
        
        if stats:
            kpi = stats['kpi_metrics']
            
            elements.append(Paragraph('Key Performance Indicators', heading_style))
            
            kpi_data = [
                ['Metric', 'Value'],
                ['Total Predictions', str(kpi['total_predictions'])],
                ['Average Likelihood', f"{kpi['average_likelihood']:.1f}%"],
                ['At-Risk Students', str(kpi['at_risk_students'])],
                ['Likely to Pass', str(kpi['likely_to_pass'])],
                ['Pass Rate', f"{kpi['pass_rate']:.1f}%"]
            ]
            
            kpi_table = Table(kpi_data, colWidths=[3*inch, 2*inch])
            kpi_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(kpi_table)
            elements.append(Spacer(1, 0.3*inch))
            
            elements.append(Paragraph('Regression Model Performance', heading_style))
            perf = stats['model_performance']
            
            perf_data = [
                ['Metric', 'Score'],
                ['Model Name', perf['model_name']],
                ['Model Type', perf['model_type']],
                ['RMSE', f"{perf['rmse']:.4f}"],
                ['MAE', f"{perf['mae']:.4f}"],
                ['R² Score', f"{perf['r2_score']:.4f}"],
                ['MSE', f"{perf['mse']:.4f}"],
            ]
            
            if perf['cv_rmse']:
                perf_data.append(['CV RMSE', f"{perf['cv_rmse']:.4f}"])
            if perf['cv_std']:
                perf_data.append(['CV Std', f"{perf['cv_std']:.4f}"])
            
            perf_table = Table(perf_data, colWidths=[3*inch, 2*inch])
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(perf_table)
            elements.append(Spacer(1, 0.3*inch))
            
            elements.append(Paragraph('User Statistics', heading_style))
            user_stats = stats['user_statistics']
            
            user_data = [
                ['Metric', 'Value'],
                ['Total Students', str(user_stats['total_students'])],
                ['Average GPA', f"{user_stats['average_gpa']:.2f}"],
                ['Average Internship Grade', f"{user_stats['average_internship_grade']:.2f}"],
                ['Average Study Hours', f"{user_stats['average_study_hours']:.1f}"],
                ['Common Study Hours', user_stats['common_study_hours']],
                ['Average Sleep Hours', f"{user_stats['average_sleep_hours']:.1f}"],
                ['Average Age', f"{user_stats['average_age']:.1f}"],
                ['Review Center Rate', f"{user_stats['review_center_rate']:.1f}%"],
                ['Scholarship Rate', f"{user_stats['scholarship_rate']:.1f}%"],
                ['Average Confidence', f"{user_stats['average_confidence']:.1f}"],
                ['Average Test Anxiety', f"{user_stats['average_test_anxiety']:.1f}"],
            ]
            
            user_table = Table(user_data, colWidths=[3*inch, 2*inch])
            user_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(user_table)
        
        doc.build(elements)
        
        pdf = buffer.getvalue()
        buffer.close()
        
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="dashboard_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf"'
        response.write(pdf)
        
        return response
        
    except ImportError:
        return JsonResponse({
            'error': 'ReportLab not installed',
            'message': 'Please install reportlab: pip install reportlab'
        }, status=500)
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'message': 'Failed to generate PDF'
        }, status=500)