from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.db import models
from .forms import PredictionForm
from .models import Prediction, PredictionHistory
from .utils import prepare_input_data, get_risk_level, generate_recommendations
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / 'models' / 'saved_base_models'


@login_required
def predict_view(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        
        if form.is_valid():
            try:
                input_data = prepare_input_data(form.cleaned_data)
                
                model_path = MODELS_DIR / 'random_forest_model.pkl'
                preprocessor_path = MODELS_DIR / 'preprocessor.pkl'
                
                if not model_path.exists():
                    messages.error(request, "Model file not found. Please contact administrator.")
                    return render(request, 'pages/predict.html', {'form': form})
                
                model = joblib.load(model_path)
                preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else None
                
                result = make_single_prediction(model, preprocessor, input_data)
                
                if result is None:
                    messages.error(request, "Error making prediction. Please try again.")
                    return render(request, 'pages/predict.html', {'form': form})
                
                predicted_class = result['prediction']
                predicted_score = result['predicted_score']
                pass_probability = result['pass_probability']
                
                prediction_obj = Prediction.objects.create(
                    user=request.user,
                    age=input_data['Age'],
                    gender=input_data['Gender'],
                    gpa=input_data['GPA'],
                    internship_grade=input_data['InternshipGrade'],
                    study_hours=input_data['StudyHours'],
                    sleep_hours=input_data['SleepHours'],
                    review_center=bool(input_data['ReviewCenter']),
                    confidence=input_data['Confidence'],
                    test_anxiety=input_data['TestAnxiety'],
                    mock_exam_score=input_data.get('MockExamScore'),
                    scholarship=bool(input_data['Scholarship']),
                    income_level=input_data['IncomeLevel'],
                    employment_status=input_data['EmploymentStatus'],
                    prediction_result='PASS' if predicted_class == 1 else 'FAIL',
                    probability=pass_probability * 100
                )
                
                PredictionHistory.objects.create(
                    user=request.user,
                    age=input_data['Age'],
                    gender=input_data['Gender'],
                    study_hours=input_data['StudyHours'],
                    sleep_hours=input_data['SleepHours'],
                    review_center=bool(input_data['ReviewCenter']),
                    confidence=input_data['Confidence'],
                    test_anxiety=input_data['TestAnxiety'],
                    mock_exam_score=input_data.get('MockExamScore'),
                    gpa=input_data['GPA'],
                    scholarship=bool(input_data['Scholarship']),
                    internship_grade=input_data['InternshipGrade'],
                    income_level=input_data['IncomeLevel'],
                    employment_status=input_data['EmploymentStatus'],
                    avg_probability=pass_probability * 100,
                    risk_level=get_risk_level(pass_probability)['level'],
                    base_predictions={'random_forest': result},
                    ip_address=request.META.get('REMOTE_ADDR'),
                    user_agent=request.META.get('HTTP_USER_AGENT', '')
                )
                
                return redirect('prediction:result', prediction_id=prediction_obj.id)
                
            except Exception as e:
                messages.error(request, f"Error processing prediction: {str(e)}")
                return render(request, 'pages/predict.html', {'form': form})
    else:
        form = PredictionForm()
    
    return render(request, 'pages/predict.html', {'form': form})


@login_required
def result_view(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
    
    input_data = {
        'Age': prediction.age,
        'Gender': prediction.gender,
        'GPA': prediction.gpa,
        'InternshipGrade': prediction.internship_grade,
        'StudyHours': prediction.study_hours,
        'SleepHours': prediction.sleep_hours,
        'ReviewCenter': prediction.review_center,
        'Confidence': prediction.confidence,
        'TestAnxiety': prediction.test_anxiety,
        'MockExamScore': prediction.mock_exam_score,
        'Scholarship': prediction.scholarship,
        'IncomeLevel': prediction.income_level,
        'EmploymentStatus': prediction.employment_status,
    }
    
    recommendations = generate_recommendations(input_data, prediction.probability / 100)
    risk_info = get_risk_level(prediction.probability / 100)
    
    return render(request, 'pages/result.html', {
        'prediction': prediction,
        'input_data': input_data,
        'recommendations': recommendations,
        'avg_probability': prediction.probability,
        'risk_info': risk_info
    })


@login_required
def history_view(request):
    predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
    
    stats = {
        'total_predictions': predictions.count(),
        'pass_predictions': predictions.filter(prediction_result='PASS').count(),
        'fail_predictions': predictions.filter(prediction_result='FAIL').count(),
        'avg_probability': predictions.aggregate(avg=models.Avg('probability'))['avg'] or 0
    }
    
    return render(request, 'pages/history.html', {
        'predictions': predictions,
        'stats': stats
    })


@login_required
def detail_view(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
    
    input_data = {
        'Age': prediction.age,
        'Gender': prediction.gender,
        'GPA': prediction.gpa,
        'InternshipGrade': prediction.internship_grade,
        'StudyHours': prediction.study_hours,
        'SleepHours': prediction.sleep_hours,
        'ReviewCenter': prediction.review_center,
        'Confidence': prediction.confidence,
        'TestAnxiety': prediction.test_anxiety,
        'MockExamScore': prediction.mock_exam_score,
        'Scholarship': prediction.scholarship,
        'IncomeLevel': prediction.income_level,
        'EmploymentStatus': prediction.employment_status,
    }
    
    recommendations = generate_recommendations(input_data, prediction.probability / 100)
    risk_info = get_risk_level(prediction.probability / 100)
    
    return render(request, 'pages/result.html', {
        'prediction': prediction,
        'input_data': input_data,
        'recommendations': recommendations,
        'avg_probability': prediction.probability,
        'risk_info': risk_info
    })


@login_required
@require_http_methods(["POST"])
def delete_prediction(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
    prediction.delete()
    messages.success(request, "Prediction deleted successfully.")
    return redirect('prediction:history')


@require_http_methods(["POST"])
@login_required
def predict_exam_result(request):
    try:
        data = request.POST
        
        input_data = {
            'Age': int(data.get('age')),
            'Gender': data.get('gender'),
            'StudyHours': int(data.get('study_hours')),
            'SleepHours': int(data.get('sleep_hours')),
            'ReviewCenter': int(data.get('review_center', 0)),
            'Confidence': int(data.get('confidence')),
            'TestAnxiety': int(data.get('test_anxiety')),
            'MockExamScore': float(data.get('mock_exam_score')) if data.get('mock_exam_score') else None,
            'GPA': float(data.get('gpa')),
            'Scholarship': int(data.get('scholarship', 0)),
            'InternshipGrade': float(data.get('internship_grade')),
            'IncomeLevel': data.get('income_level'),
            'EmploymentStatus': data.get('employment_status')
        }
        
        model_path = MODELS_DIR / 'random_forest_model.pkl'
        preprocessor_path = MODELS_DIR / 'preprocessor.pkl'
        
        if not model_path.exists():
            return JsonResponse({'error': 'Model file not found'}, status=500)
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else None
        
        result = make_single_prediction(model, preprocessor, input_data)
        
        if result is None:
            return JsonResponse({'error': 'Prediction failed'}, status=500)
        
        return JsonResponse({
            'success': True,
            'prediction': 'PASS' if result['prediction'] == 1 else 'FAIL',
            'probability': result['pass_probability'] * 100,
            'predicted_score': result['predicted_score']
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def model_status(request):
    model_path = MODELS_DIR / 'random_forest_model.pkl'
    preprocessor_path = MODELS_DIR / 'preprocessor.pkl'
    
    return JsonResponse({
        'model_loaded': model_path.exists(),
        'preprocessor_loaded': preprocessor_path.exists(),
        'model_path': str(model_path),
        'preprocessor_path': str(preprocessor_path)
    })


def make_single_prediction(model, preprocessor, input_data):
    try:
        df_input = pd.DataFrame([input_data])
        
        categorical_columns = ['Gender', 'IncomeLevel', 'EmploymentStatus']
        numerical_columns = ['Age', 'StudyHours', 'SleepHours', 'Confidence', 'TestAnxiety',
                           'MockExamScore', 'GPA', 'Scholarship', 'InternshipGrade']
        binary_columns = ['ReviewCenter']
        
        all_feature_columns = categorical_columns + numerical_columns + binary_columns
        X_input = df_input[all_feature_columns].copy()
        
        if preprocessor is None:
            print("[ERROR] Preprocessor is None!")
            return None
        
        X_input_processed = preprocessor.transform(X_input)
        
        predicted_score = model.predict(X_input_processed)[0]
        
        PASSING_SCORE = 75.0
        prediction_class = 1 if predicted_score >= PASSING_SCORE else 0
        
        return {
            'prediction': prediction_class,
            'predicted_score': float(predicted_score),
            'pass_probability': float(predicted_score / 100.0),
            'confidence': 0.85
        }
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        return None