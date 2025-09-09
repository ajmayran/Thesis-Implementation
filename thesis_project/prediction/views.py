from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
import sys

# Add the project root and models directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_path = os.path.join(project_root, 'models')
sys.path.insert(0, project_root)
sys.path.insert(0, models_path)

try:
    from models.base_models import SocialWorkPredictorModels
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import base_models: {e}")
    print(f"Project root: {project_root}")
    print(f"Models path: {models_path}")
    print("Please ensure the models directory contains base_models.py")
    MODEL_AVAILABLE = False
    SocialWorkPredictorModels = None

# Initialize global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the predictor models"""
    global predictor
    if not MODEL_AVAILABLE:
        print("Models not available due to import error")
        return None
        
    if predictor is None:
        predictor = SocialWorkPredictorModels()
        # Try to load existing models
        models_dir = os.path.join(project_root, 'saved_models')
        if os.path.exists(models_dir):
            if not predictor.load_models(models_dir):
                print("No saved models found. Please train the models first.")
        else:
            print(f"Models directory {models_dir} does not exist. Please train the models first.")
    return predictor

def prediction_form(request):
    """Render the prediction form"""
    return render(request, 'prediction/form.html')

@csrf_exempt
def make_prediction(request):
    """Handle prediction requests"""
    if request.method == 'POST':
        try:
            # Check if models are available
            if not MODEL_AVAILABLE:
                return JsonResponse({
                    'error': 'Machine learning models are not available. Please check the server configuration.',
                    'details': 'The base_models module could not be imported.'
                }, status=500)

            # Initialize predictor
            pred = initialize_predictor()
            
            if pred is None or not hasattr(pred, 'models') or not pred.models:
                return JsonResponse({
                    'error': 'Models not trained or loaded. Please train the models first.',
                    'suggestion': 'Upload training data to /api/prediction/train/ endpoint'
                }, status=500)
            
            # Parse JSON data
            if request.content_type == 'application/json':
                data = json.loads(request.body)
            else:
                data = request.POST.dict()
            
            # Convert form data to appropriate types with validation
            try:
                processed_data = {
                    'age': int(data.get('age', 25)),
                    'gender': str(data.get('gender', 'female')).lower(),
                    'gpa': str(data.get('gpa', 'G')).upper(),
                    'major_subjects': str(data.get('major_subjects', 'G')).upper(),
                    'internship_grade': str(data.get('internship_grade', 'G')).upper(),
                    'scholarship': str(data.get('scholarship', 'no')).lower(),
                    'review_center': str(data.get('review_center', 'yes')).lower(),
                    'mock_exam_score': int(data.get('mock_exam_score', 3)),
                    'test_anxiety': int(data.get('test_anxiety', 3)),
                    'confidence': int(data.get('confidence', 3)),
                    'study_hours': int(data.get('study_hours', 6)),
                    'sleeping_hours': int(data.get('sleeping_hours', 7)),
                    'income_level': str(data.get('income_level', 'middle')).lower(),
                    'employment_status': str(data.get('employment_status', 'unemployed')).lower(),
                    'employment_type': str(data.get('employment_type', 'regular')).lower(),
                    'parent_occup': str(data.get('parent_occup', 'skilled')).lower(),
                    'parent_income': str(data.get('parent_income', 'middle')).lower()
                }
            except (ValueError, TypeError) as e:
                return JsonResponse({
                    'error': f'Invalid input data: {str(e)}',
                    'details': 'Please check that all numeric fields contain valid numbers.'
                }, status=400)
            
            # Validate input ranges
            validation_errors = []
            if not (18 <= processed_data['age'] <= 65):
                validation_errors.append("Age must be between 18 and 65")
            if not (1 <= processed_data['mock_exam_score'] <= 5):
                validation_errors.append("Mock exam score must be between 1 and 5")
            if not (1 <= processed_data['test_anxiety'] <= 5):
                validation_errors.append("Test anxiety must be between 1 and 5")
            if not (1 <= processed_data['confidence'] <= 5):
                validation_errors.append("Confidence must be between 1 and 5")
            if not (0 <= processed_data['study_hours'] <= 24):
                validation_errors.append("Study hours must be between 0 and 24")
            if not (0 <= processed_data['sleeping_hours'] <= 24):
                validation_errors.append("Sleeping hours must be between 0 and 24")
                
            if validation_errors:
                return JsonResponse({
                    'error': 'Validation failed',
                    'details': validation_errors
                }, status=400)
            
            # Make prediction
            predictions = pred.predict_single(processed_data)
            
            if predictions is None:
                return JsonResponse({
                    'error': 'Failed to make prediction',
                    'details': 'The model encountered an error during prediction.'
                }, status=500)
            
            # Get feature importance
            feature_importance = pred.get_feature_importance()
            
            # Calculate ensemble probability (average of base models if ensemble not available)
            if 'ensemble' in predictions:
                final_prediction = predictions['ensemble']
            else:
                # Calculate average probability from base models
                probabilities = []
                for model_name in ['knn', 'decision_tree', 'random_forest']:
                    if model_name in predictions and predictions[model_name]['probability']:
                        probabilities.append(predictions[model_name]['probability'])
                
                if probabilities:
                    avg_prob = [sum(x)/len(probabilities) for x in zip(*probabilities)]
                    final_prediction = {
                        'prediction': 1 if avg_prob[1] > 0.5 else 0,
                        'probability': avg_prob
                    }
                else:
                    final_prediction = predictions.get('random_forest', predictions[list(predictions.keys())[0]])
            
            # Calculate confidence level
            pass_prob = final_prediction['probability'][1] if final_prediction['probability'] else 0.5
            confidence_level = 'High' if abs(pass_prob - 0.5) > 0.3 else 'Moderate' if abs(pass_prob - 0.5) > 0.15 else 'Low'
            
            # Format response
            response_data = {
                'success': True,
                'prediction': {
                    'will_pass': bool(final_prediction['prediction']),
                    'probability_pass': round(pass_prob * 100, 1),
                    'probability_fail': round((1 - pass_prob) * 100, 1),
                    'confidence_level': confidence_level,
                    'risk_level': 'Low' if pass_prob > 0.7 else 'Medium' if pass_prob > 0.4 else 'High'
                },
                'model_predictions': {
                    name: {
                        'will_pass': bool(pred_data['prediction']),
                        'probability_pass': round(pred_data['probability'][1] * 100, 1) if pred_data['probability'] else None,
                        'probability_fail': round(pred_data['probability'][0] * 100, 1) if pred_data['probability'] else None
                    } for name, pred_data in predictions.items()
                },
                'feature_importance': feature_importance,
                'recommendations': generate_recommendations(processed_data, final_prediction),
                'input_data': processed_data  # Echo back the processed input
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({
                'error': f'Prediction error: {str(e)}',
                'details': 'An unexpected error occurred during prediction.',
                'type': type(e).__name__
            }, status=500)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)

def generate_recommendations(input_data, prediction):
    """Generate personalized recommendations based on input data and prediction"""
    recommendations = []
    
    # GPA-based recommendations
    if input_data['gpa'] in ['S', 'P']:  # Satisfactory or Passing
        recommendations.append({
            'category': 'Academic Performance',
            'priority': 'high',
            'message': 'Your GPA indicates room for improvement in foundational knowledge.',
            'action': 'Focus on reviewing core social work concepts and consider academic support.',
            'resources': [
                'Study group formation',
                'Academic tutoring services',
                'Professor office hours'
            ]
        })
    
    # Study hours recommendations
    if input_data['study_hours'] < 6:
        recommendations.append({
            'category': 'Study Habits',
            'priority': 'high',
            'message': 'Increase your daily study hours for better exam preparation.',
            'action': 'Aim for 6-8 hours of focused study per day with regular breaks.',
            'resources': [
                'Pomodoro Technique',
                'Study schedule templates',
                'Time management apps'
            ]
        })
    
    # Test anxiety recommendations
    if input_data['test_anxiety'] >= 4:
        recommendations.append({
            'category': 'Test Anxiety Management',
            'priority': 'medium',
            'message': 'High anxiety levels may negatively impact your exam performance.',
            'action': 'Practice stress management and relaxation techniques regularly.',
            'resources': [
                'Breathing exercises',
                'Meditation apps',
                'Mock exam practice',
                'Counseling services'
            ]
        })
    
    # Review center recommendations
    if input_data['review_center'] == 'no':
        recommendations.append({
            'category': 'Exam Preparation',
            'priority': 'high',
            'message': 'Consider enrolling in a structured review program.',
            'action': 'Research and join a reputable review center with good passing rates.',
            'resources': [
                'Review center comparisons',
                'Online review courses',
                'Study materials'
            ]
        })
    
    # Confidence-based recommendations
    if input_data['confidence'] <= 2:
        recommendations.append({
            'category': 'Confidence Building',
            'priority': 'medium',
            'message': 'Build confidence through consistent practice and preparation.',
            'action': 'Take regular practice tests and track your improvement.',
            'resources': [
                'Practice exams',
                'Self-assessment tools',
                'Positive affirmations',
                'Study buddy system'
            ]
        })
    
    # Mock exam score recommendations
    if input_data['mock_exam_score'] <= 2:
        recommendations.append({
            'category': 'Practice Test Performance',
            'priority': 'high',
            'message': 'Your mock exam scores indicate need for more targeted practice.',
            'action': 'Focus on identifying and strengthening weak subject areas.',
            'resources': [
                'Subject-specific practice tests',
                'Error analysis worksheets',
                'Remedial study materials'
            ]
        })
    
    # Sleep recommendations
    if input_data['sleeping_hours'] < 6 or input_data['sleeping_hours'] > 9:
        recommendations.append({
            'category': 'Health & Wellness',
            'priority': 'medium',
            'message': 'Maintain optimal sleep patterns for better cognitive performance.',
            'action': 'Aim for 7-8 hours of quality sleep per night.',
            'resources': [
                'Sleep hygiene tips',
                'Relaxation techniques',
                'Sleep tracking apps'
            ]
        })
    
    # Sort recommendations by priority
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 2))
    
    return recommendations

def get_prediction_stats(request):
    """Get prediction statistics for dashboard"""
    try:
        # This would typically come from a database
        # For now, return mock data that matches your dashboard
        stats = {
            'total_predictions': 2000,
            'average_likelihood': 72.5,
            'at_risk_students': 560,
            'likely_to_pass': 1440,
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
            'user_stats': {
                'average_gpa': 2.51,
                'common_study_hours': '10 - 15',
                'average_sleeping_hours': 6.8,
                'common_review_center': 'Yes'
            },
            'model_status': {
                'available': MODEL_AVAILABLE,
                'last_trained': 'Not available' if not MODEL_AVAILABLE else 'Recently'
            }
        }
        
        return JsonResponse(stats)
        
    except Exception as e:
        return JsonResponse({
            'error': f'Stats error: {str(e)}'
        }, status=500)

@csrf_exempt
def train_models_view(request):
    """Train models with uploaded CSV data"""
    if request.method == 'POST':
        try:
            # Check if models are available
            if not MODEL_AVAILABLE:
                return JsonResponse({
                    'error': 'Machine learning models are not available. Please check the server configuration.'
                }, status=500)

            # Initialize predictor
            pred = SocialWorkPredictorModels()
            
            # Get CSV file from request
            csv_file = request.FILES.get('csv_file')
            
            if not csv_file:
                return JsonResponse({
                    'error': 'No CSV file provided',
                    'details': 'Please upload a CSV file with training data.'
                }, status=400)
            
            # Validate file type
            if not csv_file.name.endswith('.csv'):
                return JsonResponse({
                    'error': 'Invalid file type',
                    'details': 'Please upload a CSV file.'
                }, status=400)
            
            # Check file size (limit to 10MB)
            if csv_file.size > 10 * 1024 * 1024:
                return JsonResponse({
                    'error': 'File too large',
                    'details': 'Please upload a file smaller than 10MB.'
                }, status=400)
            
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                for chunk in csv_file.chunks():
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            
            try:
                # Load and preprocess data
                X, y = pred.load_and_preprocess_data(tmp_file_path)
                
                if X is None or y is None:
                    return JsonResponse({
                        'error': 'Failed to load or preprocess data',
                        'details': 'Please check that your CSV file has the correct format and columns.'
                    }, status=400)
                
                # Check minimum data requirements
                if len(X) < 50:
                    return JsonResponse({
                        'error': 'Insufficient data',
                        'details': f'At least 50 samples required for training, but only {len(X)} provided.'
                    }, status=400)
                
                # Train base models
                print("Training base models...")
                base_results = pred.train_base_models(X, y)
                
                # Create ensemble model
                print("Creating ensemble models...")
                ensemble_results, best_ensemble = pred.create_ensemble_model(X, y)
                
                # Save models
                models_dir = os.path.join(project_root, 'saved_models')
                os.makedirs(models_dir, exist_ok=True)
                pred.save_models(models_dir)
                
                # Update global predictor
                global predictor
                predictor = pred
                
                return JsonResponse({
                    'success': True,
                    'message': 'Models trained successfully',
                    'data_info': {
                        'samples': len(X),
                        'features': len(X.columns),
                        'feature_names': list(X.columns)
                    },
                    'base_results': {
                        name: {
                            'accuracy': round(results['accuracy'], 4),
                            'cv_mean': round(results['cv_mean'], 4),
                            'cv_std': round(results['cv_std'], 4)
                        } for name, results in base_results.items()
                    },
                    'ensemble_results': {
                        name: {
                            'accuracy': round(results['accuracy'], 4),
                            'cv_mean': round(results['cv_mean'], 4),
                            'cv_std': round(results['cv_std'], 4)
                        } for name, results in ensemble_results.items()
                    },
                    'best_ensemble': best_ensemble
                })
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                
        except Exception as e:
            return JsonResponse({
                'error': f'Training error: {str(e)}',
                'details': 'An unexpected error occurred during model training.',
                'type': type(e).__name__
            }, status=500)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)

def model_status(request):
    """Get current model status"""
    try:
        models_dir = os.path.join(project_root, 'saved_models')
        
        status = {
            'models_available': MODEL_AVAILABLE,
            'models_trained': False,
            'saved_models_exist': os.path.exists(models_dir),
            'model_files': []
        }
        
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            status['model_files'] = model_files
            status['models_trained'] = len(model_files) > 0
        
        if MODEL_AVAILABLE and predictor is not None:
            status['predictor_loaded'] = hasattr(predictor, 'models') and bool(predictor.models)
        
        return JsonResponse(status)
        
    except Exception as e:
        return JsonResponse({
            'error': f'Status check error: {str(e)}'
        }, status=500)