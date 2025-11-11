from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import PredictionForm
from .models import Prediction, PredictionHistory
import json
import sys
import os
import traceback
import pandas as pd

models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
sys.path.append(models_path)

from .utils import load_selected_model, prepare_input_data, generate_recommendations, get_risk_level

SELECTED_MODEL_NAME = 'random_forest'
SELECTED_MODEL_TYPE = 'base'

@login_required
def predict_view(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        
        if form.is_valid():
            form_data = form.cleaned_data
            input_data = prepare_input_data(form_data)
            
            model, preprocessor, error = load_selected_model(
                model_name=SELECTED_MODEL_NAME,
                model_type=SELECTED_MODEL_TYPE
            )
            
            if error:
                messages.error(request, error)
                return render(request, 'pages/predict.html', {'form': form})
            
            try:
                prediction_result = make_single_prediction(
                    model=model,
                    preprocessor=preprocessor,
                    input_data=input_data
                )
                
                if not prediction_result:
                    messages.error(request, 'Failed to make prediction. Please try again.')
                    return render(request, 'pages/predict.html', {'form': form})
                
                predicted_class = prediction_result['prediction']
                pass_probability = prediction_result['pass_probability']
                
                recommendations = generate_recommendations(input_data, pass_probability)
                risk_info = get_risk_level(pass_probability)
                
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
                    mock_exam_score=input_data.get('MockExamScore'),
                    gpa=input_data['GPA'],
                    scholarship=bool(input_data['Scholarship']),
                    internship_grade=input_data['InternshipGrade'],
                    income_level=input_data['IncomeLevel'],
                    employment_status=input_data['EmploymentStatus'],
                    avg_probability=pass_probability * 100,
                    risk_level=risk_info['level'],
                    base_predictions={'selected_model': prediction_result},
                    ensemble_predictions=None,
                    ip_address=request.META.get('REMOTE_ADDR'),
                    user_agent=request.META.get('HTTP_USER_AGENT')
                )
                
                messages.success(request, 'Prediction completed successfully!')
                return redirect('prediction:result', prediction_id=prediction_obj.id)
                
            except Exception as e:
                messages.error(request, f'Prediction error: {str(e)}')
                traceback.print_exc()
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
def history(request):
    predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'pages/history.html', {'predictions': predictions})

@login_required
def detail_view(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
    return redirect('prediction:result', prediction_id=prediction.id)

@login_required
@require_http_methods(["POST"])
def delete_prediction(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
    prediction.delete()
    messages.success(request, 'Prediction deleted successfully.')
    return redirect('prediction:history')

@csrf_exempt
@require_http_methods(["POST"])
def predict_exam_result(request):
    try:
        data = json.loads(request.body)
        
        model, preprocessor, error = load_selected_model(
            model_name=SELECTED_MODEL_NAME,
            model_type=SELECTED_MODEL_TYPE
        )
        
        if error:
            return JsonResponse({'error': error}, status=400)
        
        prediction_result = make_single_prediction(
            model=model,
            preprocessor=preprocessor,
            input_data=data
        )
        
        if not prediction_result:
            return JsonResponse({'error': 'Failed to make prediction'}, status=500)
        
        recommendations = generate_recommendations(data, prediction_result['pass_probability'])
        
        return JsonResponse({
            'model_name': SELECTED_MODEL_NAME,
            'model_type': SELECTED_MODEL_TYPE,
            'prediction': prediction_result,
            'recommendations': recommendations,
            'input_data': data
        })
        
    except Exception as e:
        return JsonResponse({'error': f'Prediction error: {str(e)}'}, status=500)

@require_http_methods(["GET"])
def model_status(request):
    try:
        model, preprocessor, error = load_selected_model(
            model_name=SELECTED_MODEL_NAME,
            model_type=SELECTED_MODEL_TYPE
        )
        
        if error:
            return JsonResponse({
                'model_loaded': False,
                'error': error
            })
        
        return JsonResponse({
            'model_loaded': True,
            'selected_model': SELECTED_MODEL_NAME,
            'model_type': SELECTED_MODEL_TYPE
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def make_single_prediction(model, preprocessor, input_data):
    try:
        df_input = pd.DataFrame([input_data])
        
        categorical_columns = ['Gender', 'IncomeLevel', 'EmploymentStatus']
        numerical_columns = ['Age', 'StudyHours', 'SleepHours', 'Confidence', 
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
            'prediction_label': 'Pass' if prediction_class == 1 else 'Fail'
        }
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        traceback.print_exc()
        return None