from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg, Count, Q, Max, Min
from django.db.models.functions import TruncMonth, TruncWeek, TruncYear, ExtractMonth, ExtractYear, ExtractWeek, TruncDate
from prediction.models import Prediction, RegressionModelPerformance
from .models import DashboardStatistics, MonthlyTrend
from django.utils import timezone
import json
import csv
import os
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
    
    # Get the project root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    json_path = os.path.join(
        base_dir,
        'models',
        'regression_processed_data',
        'feature_importance_analysis.json'
    )
    
    print(f"[DEBUG] Looking for feature importance at: {json_path}")
    
    try:
        if not os.path.exists(json_path):
            print(f"[WARNING] Feature importance file not found at: {json_path}")
            return get_default_feature_importance()
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print("[DEBUG] Successfully loaded feature importance JSON")
        
        # Use combined ranking average rank (lower is better)
        combined = data.get('combined_ranking', [])
        
        if not combined:
            print("[WARNING] No combined_ranking found in JSON")
            return get_default_feature_importance()
        
        # Convert to importance scores (inverse of rank, normalized)
        feature_importance = {}
        max_rank = max(item['Avg_Rank'] for item in combined)
        
        for item in combined:
            # Inverse ranking: lower rank = higher importance
            importance = (max_rank - item['Avg_Rank'] + 1) / max_rank
            feature_importance[item['Feature']] = round(importance, 4)
        
        print(f"[DEBUG] Processed {len(feature_importance)} features")
        return feature_importance
        
    except Exception as e:
        print(f"[ERROR] Error loading feature importance: {e}")
        import traceback
        traceback.print_exc()
        return get_default_feature_importance()

def get_default_feature_importance():
    """Return default feature importance values"""
    return {
        'Age': 0.85,
        'TestAnxiety': 0.72,
        'SleepHours': 0.68,
        'StudyHours': 0.58,
        'GPA': 0.52,
        'EnglishProficiency': 0.48,
        'ReviewCenter': 0.35,
        'Gender': 0.28
    }

def calculate_dashboard_statistics():
    """Calculate all dashboard statistics"""
    
    print("[DEBUG] Starting calculate_dashboard_statistics")
    
    all_predictions = Prediction.objects.all()
    total_predictions = all_predictions.count()
    
    print(f"[DEBUG] Total predictions: {total_predictions}")
    
    if total_predictions == 0:
        print("[DEBUG] No predictions found, returning None")
        return None
    
    # Calculate KPI metrics
    avg_probability = all_predictions.aggregate(Avg('probability'))['probability__avg'] or 0
    at_risk_students = all_predictions.filter(probability__lt=50).values('user').distinct().count()
    likely_to_pass = all_predictions.filter(probability__gte=75).values('user').distinct().count()
    pass_rate = (all_predictions.filter(probability__gte=50).count() / total_predictions * 100) if total_predictions > 0 else 0
    
    print(f"[DEBUG] KPI calculated - Avg prob: {avg_probability}, At risk: {at_risk_students}")
    
    # Load feature importance from JSON file
    feature_importance = load_feature_importance_from_json()
    print(f"[DEBUG] Feature importance loaded: {len(feature_importance)} features")
    
    # Get latest regression model performance
    latest_model = RegressionModelPerformance.objects.filter(
        is_active=True
    ).order_by('-trained_at').first()
    
    print(f"[DEBUG] Latest model query result: {latest_model}")
    
    if latest_model:
        print(f"[DEBUG] Model found: {latest_model.model_name}")
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
        print(f"[DEBUG] Model performance: {model_performance}")
    else:
        print("[DEBUG] No model found in database")
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
    
    # Review center and scholarship statistics
    review_center_count = all_predictions.filter(review_center=True).count()
    review_center_rate = (review_center_count / total_predictions * 100) if total_predictions > 0 else 0
    
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
    
    result = {
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
    
    print(f"[DEBUG] Returning statistics with {len(feature_importance)} features")
    return result

def calculate_monthly_trends():
    """Calculate monthly prediction trends"""
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

def calculate_weekly_trends():
    """Calculate weekly prediction trends for last 8 weeks"""
    now = timezone.now()
    eight_weeks_ago = now - timedelta(weeks=8)
    
    weekly_data = Prediction.objects.filter(
        created_at__gte=eight_weeks_ago
    ).annotate(
        week=TruncWeek('created_at')
    ).values('week').annotate(
        count=Count('id'),
        avg_prob=Avg('probability')
    ).order_by('week')
    
    # Generate labels for last 8 weeks
    labels = []
    values = []
    
    for i in range(8):
        week_start = now - timedelta(weeks=(7-i))
        labels.append(f'Week {i+1}')
    
    # Map data to labels
    week_dict = {}
    for data in weekly_data:
        week_num = (now - data['week']).days // 7
        if 0 <= week_num < 8:
            week_dict[7 - week_num] = round(data['avg_prob'], 1)
    
    # Fill values
    for i in range(8):
        values.append(week_dict.get(i, 0))
    
    return {
        'labels': labels,
        'values': values
    }

def calculate_yearly_trends():
    """Calculate yearly prediction trends for last 5 years"""
    now = timezone.now()
    current_year = now.year
    five_years_ago = now - timedelta(days=365*5)
    
    yearly_data = Prediction.objects.filter(
        created_at__gte=five_years_ago
    ).annotate(
        year=TruncYear('created_at')
    ).values('year').annotate(
        count=Count('id'),
        avg_prob=Avg('probability')
    ).order_by('year')
    
    # Generate labels for last 5 years
    labels = [str(current_year - 4 + i) for i in range(5)]
    values = [0] * 5
    
    # Map data to years
    for data in yearly_data:
        year = data['year'].year
        year_index = year - (current_year - 4)
        if 0 <= year_index < 5:
            values[year_index] = round(data['avg_prob'], 1)
    
    return {
        'labels': labels,
        'values': values
    }

@require_http_methods(["GET"])
@login_required
def get_trend_data(request):
    """API endpoint to get trend data based on period"""
    try:
        period = request.GET.get('period', 'monthly')
        
        print(f"[DEBUG] get_trend_data called with period: {period}")
        
        if period == 'weekly':
            trend_data = calculate_weekly_trends()
        elif period == 'monthly':
            trend_data = calculate_monthly_trends()
        elif period == 'yearly':
            trend_data = calculate_yearly_trends()
        else:
            return JsonResponse({
                'error': 'Invalid period',
                'message': 'Period must be weekly, monthly, or yearly'
            }, status=400)
        
        print(f"[DEBUG] Returning trend data: {len(trend_data['labels'])} points")
        
        return JsonResponse(trend_data)
        
    except Exception as e:
        print(f"[ERROR] get_trend_data error: {e}")
        import traceback
        traceback.print_exc()
        
        return JsonResponse({
            'error': str(e),
            'message': 'Failed to load trend data'
        }, status=500)

@require_http_methods(["GET"])
@login_required
def dashboard_stats(request):
    """API endpoint for dashboard statistics"""
    try:
        print("[DEBUG] dashboard_stats endpoint called")
        
        stats = calculate_dashboard_statistics()
        
        if not stats:
            print("[DEBUG] No stats available, returning empty data")
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
        
        print(f"[DEBUG] Returning stats with {len(stats['feature_importance'])} features")
        
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
        print(f"[ERROR] dashboard_stats error: {e}")
        import traceback
        traceback.print_exc()
        
        return JsonResponse({
            'error': str(e),
            'message': 'Failed to load dashboard statistics'
        }, status=500)

@require_http_methods(["GET"])
@login_required
def export_csv(request):
    """Export predictions to CSV"""
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
    """Generate PDF report"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.enums import TA_CENTER
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
        print(f"[ERROR] PDF generation error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'error': str(e),
            'message': 'Failed to generate PDF'
        }, status=500)
    
@login_required
def reports_view(request):
    """Render the reports page"""
    return render(request, 'pages/reports.html')

@require_http_methods(["GET"])
@login_required
def get_reports_data(request):
    """API endpoint for reports data with filters"""
    try:
        # Get filter parameters
        date_from = request.GET.get('dateFrom', '')
        date_to = request.GET.get('dateTo', '')
        risk_level = request.GET.get('riskLevel', 'all')
        department = request.GET.get('department', 'all')
        
        # Base queryset
        predictions = Prediction.objects.all()
        
        # Apply date filters
        if date_from:
            date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
            predictions = predictions.filter(created_at__gte=date_from_obj)
        
        if date_to:
            date_to_obj = datetime.strptime(date_to, '%Y-%m-%d')
            predictions = predictions.filter(created_at__lte=date_to_obj)
        
        # Apply risk level filter
        if risk_level == 'high':
            predictions = predictions.filter(probability__lt=50)
        elif risk_level == 'medium':
            predictions = predictions.filter(probability__gte=50, probability__lt=70)
        elif risk_level == 'low':
            predictions = predictions.filter(probability__gte=70)
        
        # Calculate overview data
        total_predictions = predictions.count()
        avg_likelihood = predictions.aggregate(Avg('probability'))['probability__avg'] or 0
        at_risk_students = predictions.filter(probability__lt=50).values('user').distinct().count()
        active_users = predictions.values('user').distinct().count()
        
        overview_data = {
            'total_predictions': total_predictions,
            'average_likelihood': round(avg_likelihood, 1),
            'at_risk_students': at_risk_students,
            'active_users': active_users
        }
        
        # Calculate trend data (last 30 days)
        thirty_days_ago = timezone.now() - timedelta(days=30)
        trend_predictions = predictions.filter(created_at__gte=thirty_days_ago).annotate(
            date=TruncDate('created_at')
        ).values('date').annotate(
            avg_prob=Avg('probability')
        ).order_by('date')
        
        trend_labels = []
        trend_values = []
        for item in trend_predictions:
            trend_labels.append(item['date'].strftime('%m/%d'))
            trend_values.append(round(item['avg_prob'], 1))
        
        trend_data = {
            'labels': trend_labels,
            'values': trend_values
        }
        
        # Risk distribution
        high_risk = predictions.filter(probability__lt=50).count()
        medium_risk = predictions.filter(probability__gte=50, probability__lt=70).count()
        low_risk = predictions.filter(probability__gte=70).count()
        
        risk_distribution = {
            'high': high_risk,
            'medium': medium_risk,
            'low': low_risk
        }
        
        # Performance metrics
        latest_model = RegressionModelPerformance.objects.filter(
            is_active=True
        ).order_by('-trained_at').first()
        
        if latest_model:
            performance_data = {
                'rmse': round(latest_model.rmse, 4),
                'mae': round(latest_model.mae, 4),
                'r2_score': round(latest_model.r2_score, 4),
                'mse': round(latest_model.mse, 4),
                'model_name': latest_model.model_name,
                'model_type': latest_model.model_type,
                'trained_at': latest_model.trained_at.strftime('%Y-%m-%d %H:%M:%S'),
                'total_predictions': total_predictions
            }
        else:
            performance_data = {
                'rmse': 0,
                'mae': 0,
                'r2_score': 0,
                'mse': 0,
                'model_name': 'No Model',
                'model_type': 'N/A',
                'trained_at': 'Not trained yet',
                'total_predictions': 0
            }
        
        # Feature importance
        feature_importance = load_feature_importance_from_json()
        
        # Accuracy trend (last 10 model trainings)
        model_history = RegressionModelPerformance.objects.order_by('-trained_at')[:10]
        accuracy_labels = []
        accuracy_values = []
        for model in reversed(list(model_history)):
            accuracy_labels.append(model.trained_at.strftime('%m/%d'))
            accuracy_values.append(round(model.r2_score, 4))
        
        accuracy_trend = {
            'labels': accuracy_labels,
            'values': accuracy_values
        }
        
        # Recent predictions for predictions report
        recent_predictions = predictions.select_related('user').order_by('-created_at')[:50]
        predictions_list = []
        for pred in recent_predictions:
            predictions_list.append({
                'student_name': pred.user.get_full_name(),
                'student_id': pred.user.student_id or 'N/A',
                'likelihood': round(pred.probability, 1),
                'prediction_date': pred.created_at.strftime('%Y-%m-%d')
            })
        
        # Likelihood ranges for distribution
        ranges = {
            '0-20': predictions.filter(probability__lt=20).count(),
            '20-40': predictions.filter(probability__gte=20, probability__lt=40).count(),
            '40-60': predictions.filter(probability__gte=40, probability__lt=60).count(),
            '60-80': predictions.filter(probability__gte=60, probability__lt=80).count(),
            '80-100': predictions.filter(probability__gte=80).count()
        }
        
        # Risk analysis data
        risk_analysis = {
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'total_assessed': total_predictions
        }
        
        # Risk factors (based on feature importance)
        risk_factors = {}
        for feature, importance in feature_importance.items():
            risk_factors[feature] = importance
        
        # Risk trends over time (last 6 months)
        six_months_ago = timezone.now() - timedelta(days=180)
        risk_trends_data = predictions.filter(created_at__gte=six_months_ago).annotate(
            month=TruncMonth('created_at')
        ).values('month').annotate(
            high=Count('id', filter=Q(probability__lt=50)),
            medium=Count('id', filter=Q(probability__gte=50, probability__lt=70)),
            low=Count('id', filter=Q(probability__gte=70))
        ).order_by('month')
        
        risk_trends = {
            'labels': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        for item in risk_trends_data:
            risk_trends['labels'].append(item['month'].strftime('%b %Y'))
            risk_trends['high'].append(item['high'])
            risk_trends['medium'].append(item['medium'])
            risk_trends['low'].append(item['low'])
        
        # Compile all data
        report_data = {
            'overview': overview_data,
            'trend_data': trend_data,
            'risk_distribution': risk_distribution,
            'performance': performance_data,
            'feature_importance': feature_importance,
            'accuracy_trend': accuracy_trend,
            'predictions': predictions_list,
            'likelihood_ranges': ranges,
            'risk_analysis': risk_analysis,
            'risk_factors': risk_factors,
            'risk_trends': risk_trends
        }
        
        return JsonResponse(report_data)
        
    except Exception as e:
        print(f"[ERROR] get_reports_data error: {e}")
        import traceback
        traceback.print_exc()
        
        return JsonResponse({
            'error': str(e),
            'message': 'Failed to load report data'
        }, status=500)

@require_http_methods(["GET"])
@login_required
def export_reports_pdf(request):
    """Export reports to PDF"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from io import BytesIO
        
        report_type = request.GET.get('reportType', 'overview')
        date_from = request.GET.get('dateFrom', '')
        date_to = request.GET.get('dateTo', '')
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1f2937'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        elements.append(Paragraph('CPA Licensure Exam Predictor', title_style))
        elements.append(Paragraph(f'{report_type.title()} Report', styles['Heading2']))
        elements.append(Paragraph(f'Generated: {datetime.now().strftime("%B %d, %Y %H:%M:%S")}', styles['Normal']))
        
        if date_from and date_to:
            elements.append(Paragraph(f'Period: {date_from} to {date_to}', styles['Normal']))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Get data
        predictions = Prediction.objects.all()
        if date_from:
            predictions = predictions.filter(created_at__gte=datetime.strptime(date_from, '%Y-%m-%d'))
        if date_to:
            predictions = predictions.filter(created_at__lte=datetime.strptime(date_to, '%Y-%m-%d'))
        
        # Generate report content based on type
        if report_type == 'overview':
            # Overview statistics
            elements.append(Paragraph('System Overview', heading_style))
            
            overview_data = [
                ['Metric', 'Value'],
                ['Total Predictions', str(predictions.count())],
                ['Average Likelihood', f"{predictions.aggregate(Avg('probability'))['probability__avg'] or 0:.1f}%"],
                ['At Risk Students', str(predictions.filter(probability__lt=50).values('user').distinct().count())],
                ['Active Users', str(predictions.values('user').distinct().count())]
            ]
            
            overview_table = Table(overview_data, colWidths=[3*inch, 2*inch])
            overview_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(overview_table)
        
        elif report_type == 'performance':
            # Model performance
            elements.append(Paragraph('Model Performance Metrics', heading_style))
            
            latest_model = RegressionModelPerformance.objects.filter(is_active=True).order_by('-trained_at').first()
            
            if latest_model:
                perf_data = [
                    ['Metric', 'Value'],
                    ['Model Name', latest_model.model_name],
                    ['Model Type', latest_model.model_type],
                    ['RMSE', f"{latest_model.rmse:.4f}"],
                    ['MAE', f"{latest_model.mae:.4f}"],
                    ['R² Score', f"{latest_model.r2_score:.4f}"],
                    ['MSE', f"{latest_model.mse:.4f}"],
                    ['Trained At', latest_model.trained_at.strftime('%Y-%m-%d %H:%M:%S')]
                ]
                
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
        
        elif report_type == 'predictions':
            # Predictions list
            elements.append(Paragraph('Recent Predictions', heading_style))
            
            pred_data = [['No.', 'Student', 'ID', 'Likelihood', 'Risk', 'Date']]
            
            recent_preds = predictions.select_related('user').order_by('-created_at')[:20]
            for idx, pred in enumerate(recent_preds, 1):
                risk = 'High' if pred.probability < 50 else 'Medium' if pred.probability < 70 else 'Low'
                pred_data.append([
                    str(idx),
                    pred.user.get_full_name(),
                    pred.user.student_id or 'N/A',
                    f"{pred.probability:.1f}%",
                    risk,
                    pred.created_at.strftime('%Y-%m-%d')
                ])
            
            pred_table = Table(pred_data, colWidths=[0.5*inch, 1.5*inch, 1*inch, 1*inch, 0.8*inch, 1*inch])
            pred_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))
            
            elements.append(pred_table)
        
        elif report_type == 'risk':
            # Risk analysis
            elements.append(Paragraph('Risk Analysis', heading_style))
            
            high_risk = predictions.filter(probability__lt=50).count()
            medium_risk = predictions.filter(probability__gte=50, probability__lt=70).count()
            low_risk = predictions.filter(probability__gte=70).count()
            
            risk_data = [
                ['Risk Level', 'Count', 'Percentage'],
                ['High Risk', str(high_risk), f"{(high_risk/predictions.count()*100) if predictions.count() > 0 else 0:.1f}%"],
                ['Medium Risk', str(medium_risk), f"{(medium_risk/predictions.count()*100) if predictions.count() > 0 else 0:.1f}%"],
                ['Low Risk', str(low_risk), f"{(low_risk/predictions.count()*100) if predictions.count() > 0 else 0:.1f}%"],
                ['Total', str(predictions.count()), '100.0%']
            ]
            
            risk_table = Table(risk_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ef4444')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(risk_table)
        
        # Build PDF
        doc.build(elements)
        
        pdf = buffer.getvalue()
        buffer.close()
        
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="report_{report_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf"'
        response.write(pdf)
        
        return response
        
    except ImportError:
        return JsonResponse({
            'error': 'ReportLab not installed',
            'message': 'Please install reportlab: pip install reportlab'
        }, status=500)
    except Exception as e:
        print(f"[ERROR] PDF export error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'error': str(e),
            'message': 'Failed to export PDF'
        }, status=500)

@require_http_methods(["GET"])
@login_required
def export_reports_csv(request):
    """Export reports to CSV"""
    try:
        report_type = request.GET.get('reportType', 'overview')
        date_from = request.GET.get('dateFrom', '')
        date_to = request.GET.get('dateTo', '')
        
        # Get predictions
        predictions = Prediction.objects.all()
        if date_from:
            predictions = predictions.filter(created_at__gte=datetime.strptime(date_from, '%Y-%m-%d'))
        if date_to:
            predictions = predictions.filter(created_at__lte=datetime.strptime(date_to, '%Y-%m-%d'))
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="report_{report_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
        
        writer = csv.writer(response)
        
        if report_type == 'predictions':
            writer.writerow([
                'No.', 'Student Name', 'Student ID', 'Email', 'Likelihood (%)',
                'Risk Level', 'Age', 'Gender', 'GPA', 'Study Hours',
                'Sleep Hours', 'Review Center', 'Prediction Date'
            ])
            
            for idx, pred in enumerate(predictions.select_related('user').order_by('-created_at'), 1):
                risk = 'High' if pred.probability < 50 else 'Medium' if pred.probability < 70 else 'Low'
                writer.writerow([
                    idx,
                    pred.user.get_full_name(),
                    pred.user.student_id or 'N/A',
                    pred.user.email,
                    f"{pred.probability:.1f}",
                    risk,
                    pred.age,
                    pred.gender,
                    f"{pred.gpa:.2f}",
                    pred.study_hours,
                    pred.sleep_hours,
                    'Yes' if pred.review_center else 'No',
                    pred.created_at.strftime('%Y-%m-%d %H:%M:%S')
                ])
        
        elif report_type == 'risk':
            writer.writerow(['Risk Level', 'Count', 'Percentage'])
            
            total = predictions.count()
            high_risk = predictions.filter(probability__lt=50).count()
            medium_risk = predictions.filter(probability__gte=50, probability__lt=70).count()
            low_risk = predictions.filter(probability__gte=70).count()
            
            writer.writerow(['High Risk', high_risk, f"{(high_risk/total*100) if total > 0 else 0:.1f}"])
            writer.writerow(['Medium Risk', medium_risk, f"{(medium_risk/total*100) if total > 0 else 0:.1f}"])
            writer.writerow(['Low Risk', low_risk, f"{(low_risk/total*100) if total > 0 else 0:.1f}"])
            writer.writerow(['Total', total, '100.0'])
        
        else:
            # Overview or performance
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Predictions', predictions.count()])
            writer.writerow(['Average Likelihood', f"{predictions.aggregate(Avg('probability'))['probability__avg'] or 0:.1f}"])
            writer.writerow(['At Risk Students', predictions.filter(probability__lt=50).values('user').distinct().count()])
            writer.writerow(['Active Users', predictions.values('user').distinct().count()])
        
        return response
        
    except Exception as e:
        print(f"[ERROR] CSV export error: {e}")
        return JsonResponse({
            'error': str(e),
            'message': 'Failed to export CSV'
        }, status=500)