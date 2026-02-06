from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import json
import sys
import os
import traceback
import pandas as pd
import pickle
from .models import PredictionHistory, Prediction

models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
sys.path.append(models_path)

from .forms import PredictionForm
from .utils import prepare_input_data, generate_recommendations, get_risk_level

# Configuration: Change these variables to switch models
# For base models: 'decision_tree', 'random_forest', 'knn', 'ridge', 'svr'
# For ensemble models: 'bagging_random_forest', 'boosting_gradient_boost', 'stacking_ridge'
SELECTED_MODEL_NAME = 'stacking_ridge'
MODEL_CATEGORY = 'ensemble'  

# Model paths based on directory structure
BASE_MODELS_DIR = os.path.join(models_path, 'saved_base_models')
ENSEMBLE_MODELS_DIR = os.path.join(models_path, 'saved_ensemble_models')
PREPROCESSOR_PATH = os.path.join(models_path, 'regression_processed_data', 'preprocessor.pkl')

def load_selected_model(model_name, model_category='base'):
    """
    Load the selected regression model and preprocessor
    
    Args:
        model_name: Name of the model file without extension
        model_category: 'base' or 'ensemble'
    
    Returns:
        tuple: (model, preprocessor, error_message)
    """
    try:
        # Determine model directory based on category
        if model_category == 'base':
            model_dir = BASE_MODELS_DIR
            model_file = f"{model_name}_model.pkl"
        elif model_category == 'ensemble':
            model_dir = ENSEMBLE_MODELS_DIR
            model_file = f"{model_name}_ensemble.pkl"
        else:
            return None, None, f"Invalid model category: {model_category}"
        
        model_path = os.path.join(model_dir, model_file)
        
        # Check if model file exists
        if not os.path.exists(model_path):
            return None, None, f"Model file not found: {model_path}"
        
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load the preprocessor
        if not os.path.exists(PREPROCESSOR_PATH):
            return None, None, f"Preprocessor file not found: {PREPROCESSOR_PATH}"
        
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        
        print(f"Successfully loaded {model_category} model: {model_name}")
        return model, preprocessor, None
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        return None, None, error_msg

@login_required
def predict_view(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        
        if form.is_valid():
            form_data = form.cleaned_data
            input_data = prepare_input_data(form_data)
            
            # Load the selected model
            model, preprocessor, error = load_selected_model(
                model_name=SELECTED_MODEL_NAME,
                model_category=MODEL_CATEGORY
            )
            
            if error:
                messages.error(request, error)
                return render(request, 'pages/predict.html', {'form': form})
            
            try:
                # Make prediction
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
                
                # Generate recommendations and risk assessment
                recommendations = generate_recommendations(input_data, pass_probability)
                risk_info = get_risk_level(pass_probability)
                
                # Save prediction to database
                prediction = Prediction.objects.create(
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
                
                # Save to prediction history
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
                    risk_level=risk_info['level'],
                    base_predictions={
                        'model_name': SELECTED_MODEL_NAME,
                        'model_category': MODEL_CATEGORY,
                        'prediction': prediction_result
                    },
                    ensemble_predictions=None,
                    ip_address=request.META.get('REMOTE_ADDR'),
                    user_agent=request.META.get('HTTP_USER_AGENT')
                )
                
                # Store results in session
                request.session['prediction_results'] = {
                    'input_data': input_data,
                    'prediction': prediction_result,
                    'model_name': SELECTED_MODEL_NAME,
                    'model_category': MODEL_CATEGORY,
                    'recommendations': recommendations,
                    'avg_probability': pass_probability * 100,
                    'risk_info': risk_info,
                    'prediction_id': prediction.id
                }
                
                return redirect('prediction:result', prediction_id=prediction.id)
                
            except Exception as e:
                messages.error(request, f'Prediction error: {str(e)}')
                traceback.print_exc()
                return render(request, 'pages/predict.html', {'form': form})
    else:
        form = PredictionForm()
    
    return render(request, 'pages/predict.html', {'form': form})

@login_required
def result_view(request, prediction_id):
    try:
        prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
        
        results = request.session.get('prediction_results')
        
        if results and results.get('prediction_id') == prediction_id:
            context = {
                'input_data': results['input_data'],
                'predictions': {
                    'base_models': {
                        results['model_name']: results['prediction']
                    },
                    'ensemble_models': {}
                },
                'model_name': results['model_name'],
                'model_category': results.get('model_category', MODEL_CATEGORY),
                'recommendations': results['recommendations'],
                'avg_probability': results['avg_probability'],
                'risk_info': results['risk_info'],
                'prediction': prediction
            }
        else:
            # Reconstruct data from database
            input_data = {
                'Age': prediction.age,
                'Gender': prediction.gender,
                'GPA': prediction.gpa,
                'InternshipGrade': prediction.internship_grade,
                'StudyHours': prediction.study_hours,
                'SleepHours': prediction.sleep_hours,
                'ReviewCenter': 1 if prediction.review_center else 0,
                'Confidence': prediction.confidence,
                'TestAnxiety': prediction.test_anxiety,
                'MockExamScore': prediction.mock_exam_score,
                'Scholarship': 1 if prediction.scholarship else 0,
                'IncomeLevel': prediction.income_level,
                'EmploymentStatus': prediction.employment_status
            }
            
            prediction_result = {
                'prediction': 1 if prediction.prediction_result == 'PASS' else 0,
                'pass_probability': prediction.probability / 100,
                'predicted_score': prediction.probability,
                'prediction_label': prediction.prediction_result
            }
            
            recommendations = generate_recommendations(input_data, prediction.probability / 100)
            risk_info = get_risk_level(prediction.probability / 100)
            
            context = {
                'input_data': input_data,
                'predictions': {
                    'base_models': {
                        SELECTED_MODEL_NAME: prediction_result
                    },
                    'ensemble_models': {}
                },
                'model_name': SELECTED_MODEL_NAME,
                'model_category': MODEL_CATEGORY,
                'recommendations': recommendations,
                'avg_probability': prediction.probability,
                'risk_info': risk_info,
                'prediction': prediction
            }
        
        return render(request, 'pages/result.html', context)
        
    except Prediction.DoesNotExist:
        messages.error(request, 'Prediction not found.')
        return redirect('prediction:history')

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
        'ReviewCenter': 1 if prediction.review_center else 0,
        'Confidence': prediction.confidence,
        'TestAnxiety': prediction.test_anxiety,
        'MockExamScore': prediction.mock_exam_score,
        'Scholarship': 1 if prediction.scholarship else 0,
        'IncomeLevel': prediction.income_level,
        'EmploymentStatus': prediction.employment_status
    }
    
    prediction_result = {
        'prediction': 1 if prediction.prediction_result == 'PASS' else 0,
        'pass_probability': prediction.probability / 100,
        'predicted_score': prediction.probability,
        'prediction_label': prediction.prediction_result
    }
    
    recommendations = generate_recommendations(input_data, prediction.probability / 100)
    risk_info = get_risk_level(prediction.probability / 100)
    
    context = {
        'input_data': input_data,
        'predictions': {
            'base_models': {
                SELECTED_MODEL_NAME: prediction_result
            },
            'ensemble_models': {}
        },
        'model_name': SELECTED_MODEL_NAME,
        'model_category': MODEL_CATEGORY,
        'recommendations': recommendations,
        'avg_probability': prediction.probability,
        'risk_info': risk_info,
        'prediction': prediction
    }
    
    return render(request, 'pages/result.html', context)

@login_required
def history_view(request):
    predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'predictions': predictions,
        'total_predictions': predictions.count()
    }
    
    return render(request, 'pages/history.html', context)

@login_required
@require_http_methods(["POST"])
def delete_prediction(request, prediction_id):
    try:
        prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
        prediction.delete()
        messages.success(request, 'Prediction deleted successfully.')
    except Exception as e:
        messages.error(request, f'Failed to delete prediction: {str(e)}')
    
    return redirect('prediction:history')

@csrf_exempt
@require_http_methods(["POST"])
def predict_exam_result(request):
    try:
        data = json.loads(request.body)
        
        model, preprocessor, error = load_selected_model(
            model_name=SELECTED_MODEL_NAME,
            model_category=MODEL_CATEGORY
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
            'model_category': MODEL_CATEGORY,
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
            model_category=MODEL_CATEGORY
        )
        
        if error:
            return JsonResponse({
                'model_loaded': False,
                'error': error
            })
        
        return JsonResponse({
            'model_loaded': True,
            'selected_model': SELECTED_MODEL_NAME,
            'model_category': MODEL_CATEGORY,
            'model_path': ENSEMBLE_MODELS_DIR if MODEL_CATEGORY == 'ensemble' else BASE_MODELS_DIR
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def make_single_prediction(model, preprocessor, input_data):
    """
    Make a single prediction using regression model
    
    Args:
        model: Loaded regression model
        preprocessor: Loaded preprocessor
        input_data: Dictionary containing input features
    
    Returns:
        dict: Prediction results with score and pass/fail classification
    """
    try:
        # Create DataFrame from input
        df_input = pd.DataFrame([input_data])
        
        # Define feature columns
        categorical_columns = ['Gender', 'IncomeLevel', 'EmploymentStatus']
        numerical_columns = ['Age', 'StudyHours', 'SleepHours', 'Confidence', 
                           'MockExamScore', 'GPA', 'InternshipGrade', 'TestAnxiety']
        binary_columns = ['ReviewCenter', 'Scholarship']
        
        all_feature_columns = categorical_columns + numerical_columns + binary_columns
        X_input = df_input[all_feature_columns].copy()
        
        # Handle missing MockExamScore
        if 'MockExamScore' in X_input.columns and X_input['MockExamScore'].isnull().any():
            X_input.loc[:, 'MockExamScore'] = X_input['MockExamScore'].fillna(77.0)
        
        # Preprocess the input data
        if preprocessor is None:
            print("[ERROR] Preprocessor is None")
            return None
            
        X_input_processed = preprocessor.transform(X_input)
        
        # Make prediction using regression model
        predicted_score = model.predict(X_input_processed)[0]
        
        # Convert regression score to classification
        PASSING_SCORE = 70.0
        prediction_class = 1 if predicted_score >= PASSING_SCORE else 0
        pass_probability = float(predicted_score / 100.0)
        
        # Ensure probability is within valid range
        pass_probability = max(0.0, min(1.0, pass_probability))
        
        return {
            'prediction': prediction_class,
            'predicted_score': float(predicted_score),
            'pass_probability': pass_probability,
            'prediction_label': 'PASS' if prediction_class == 1 else 'FAIL'
        }
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        traceback.print_exc()
        return None