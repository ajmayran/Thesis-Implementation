from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
import sys

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

try:
    from base_models import SocialWorkPredictorModels
except ImportError:
    print("Warning: base_models module not found. Please ensure the models directory is set up correctly.")

# Initialize global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the predictor models"""
    global predictor
    if predictor is None:
        predictor = SocialWorkPredictorModels()
        # Try to load existing models
        models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'saved_models')
        if not predictor.load_models(models_dir):
            print("No saved models found. Please train the models first.")
    return predictor

def prediction_form(request):
    """Render the prediction form"""
    return render(request, 'prediction/form.html')

@csrf_exempt
def make_prediction(request):
    """Handle prediction requests"""
    if request.method == 'POST':
        try:
            # Initialize predictor
            pred = initialize_predictor()
            
            if pred is None or not pred.models:
                return JsonResponse({
                    'error': 'Models not available. Please train the models first.'
                }, status=500)
            
            # Parse JSON data
            if request.content_type == 'application/json':
                data = json.loads(request.body)
            else:
                data = request.POST.dict()
            
            # Convert form data to appropriate types
            processed_data = {
                'age': int(data.get('age', 25)),
                'gender': data.get('gender', 'female'),
                'gpa': data.get('gpa', 'G'),
                'major_subjects': data.get('major_subjects', 'G'),
                'internship_grade': data.get('internship_grade', 'G'),
                'scholarship': data.get('scholarship', 'no'),
                'review_center': data.get('review_center', 'yes'),
                'mock_exam_score': int(data.get('mock_exam_score', 3)),
                'test_anxiety': int(data.get('test_anxiety', 3)),
                'confidence': int(data.get('confidence', 3)),
                'study_hours': int(data.get('study_hours', 6)),
                'sleeping_hours': int(data.get('sleeping_hours', 7)),
                'income_level': data.get('income_level', 'middle'),
                'employment_status': data.get('employment_status', 'unemployed'),
                'employment_type': data.get('employment_type', 'regular'),
                'parent_occup': data.get('parent_occup', 'skilled'),
                'parent_income': data.get('parent_income', 'middle')
            }
            
            # Make prediction
            predictions = pred.predict_single(processed_data)
            
            if predictions is None:
                return JsonResponse({
                    'error': 'Failed to make prediction'
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
            
            # Format response
            response_data = {
                'success': True,
                'prediction': {
                    'will_pass': bool(final_prediction['prediction']),
                    'probability_pass': final_prediction['probability'][1] if final_prediction['probability'] else 0.5,
                    'probability_fail': final_prediction['probability'][0] if final_prediction['probability'] else 0.5,
                    'confidence_level': 'High' if abs(final_prediction['probability'][1] - 0.5) > 0.3 else 'Moderate'
                },
                'model_predictions': {
                    name: {
                        'will_pass': bool(pred_data['prediction']),
                        'probability': pred_data['probability']
                    } for name, pred_data in predictions.items()
                },
                'feature_importance': feature_importance,
                'recommendations': generate_recommendations(processed_data, final_prediction)
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({
                'error': f'Prediction error: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)

def generate_recommendations(input_data, prediction):
    """Generate personalized recommendations based on input data and prediction"""
    recommendations = []
    
    # GPA-based recommendations
    if input_data['gpa'] in ['S', 'P']:  # Satisfactory or Passing
        recommendations.append({
            'category': 'Academic Performance',
            'message': 'Focus on improving your foundational knowledge in core social work subjects.',
            'action': 'Consider reviewing basic concepts and seeking additional academic support.'
        })
    
    # Study hours recommendations
    if input_data['study_hours'] < 6:
        recommendations.append({
            'category': 'Study Habits',
            'message': 'Increase your daily study hours to at least 6-8 hours for better preparation.',
            'action': 'Create a structured study schedule with regular breaks.'
        })
    
    # Test anxiety recommendations
    if input_data['test_anxiety'] >= 4:
        recommendations.append({
            'category': 'Test Anxiety',
            'message': 'Your anxiety levels are high. Consider stress management techniques.',
            'action': 'Practice relaxation techniques, take mock exams, and consider counseling support.'
        })
    
    # Review center recommendations
    if input_data['review_center'] == 'no':
        recommendations.append({
            'category': 'Exam Preparation',
            'message': 'Consider enrolling in a review center for structured preparation.',
            'action': 'Research reputable review centers with good passing rates.'
        })
    
    # Confidence-based recommendations
    if input_data['confidence'] <= 2:
        recommendations.append({
            'category': 'Confidence Building',
            'message': 'Work on building your confidence through consistent practice.',
            'action': 'Take practice tests regularly and celebrate small achievements.'
        })
    
    # Mock exam score recommendations
    if input_data['mock_exam_score'] <= 2:
        recommendations.append({
            'category': 'Practice Tests',
            'message': 'Your mock exam scores suggest need for more practice.',
            'action': 'Take more practice exams and focus on weak areas identified.'
        })
    
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
            }
        }
        
        return JsonResponse(stats)
        
    except Exception as e:
        return JsonResponse({
            'error': f'Stats error: {str(e)}'
        }, status=500)

def train_models_view(request):
    """Train models with uploaded CSV data"""
    if request.method == 'POST':
        try:
            # Initialize predictor
            pred = SocialWorkPredictorModels()
            
            # Get CSV file from request
            csv_file = request.FILES.get('csv_file')
            
            if not csv_file:
                return JsonResponse({
                    'error': 'No CSV file provided'
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
                        'error': 'Failed to load or preprocess data'
                    }, status=400)
                
                # Train base models
                base_results = pred.train_base_models(X, y)
                
                # Create ensemble model
                ensemble_results, best_ensemble = pred.create_ensemble_model(X, y)
                
                # Save models
                models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'saved_models')
                pred.save_models(models_dir)
                
                # Update global predictor
                global predictor
                predictor = pred
                
                return JsonResponse({
                    'success': True,
                    'message': 'Models trained successfully',
                    'base_results': {
                        name: {
                            'accuracy': results['accuracy'],
                            'cv_mean': results['cv_mean'],
                            'cv_std': results['cv_std']
                        } for name, results in base_results.items()
                    },
                    'ensemble_results': {
                        name: {
                            'accuracy': results['accuracy'],
                            'cv_mean': results['cv_mean'],
                            'cv_std': results['cv_std']
                        } for name, results in ensemble_results.items()
                    },
                    'best_ensemble': best_ensemble
                })
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            return JsonResponse({
                'error': f'Training error: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)