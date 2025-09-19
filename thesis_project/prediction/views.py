from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import sys
import os

# Add the models directory to the Python path
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
sys.path.append(models_path)

@csrf_exempt
@require_http_methods(["POST"])
def predict_exam_result(request):
    """Handle prediction requests"""
    try:
        # Parse JSON data
        data = json.loads(request.body)
        
        # Import and use your models
        from models.base_models import SocialWorkPredictorModels
        from models.ensembles import EnsembleModels
        
        # Initialize predictor
        predictor = SocialWorkPredictorModels()
        
        # Try to load saved models
        models_loaded = predictor.load_models('saved_base_models')
        
        if not models_loaded or not predictor.models:
            return JsonResponse({
                'error': 'Models not trained. Please train models first.',
                'suggestion': 'Go to admin panel to train models with your data.'
            }, status=400)
        
        # Make predictions with base models
        base_predictions = predictor.predict_single(data)
        
        if not base_predictions:
            return JsonResponse({
                'error': 'Failed to make base model predictions'
            }, status=500)
        
        # Try ensemble predictions if available
        ensemble_predictions = {}
        try:
            ensemble = EnsembleModels(predictor)
            if ensemble.load_ensembles('saved_ensemble_models'):
                all_predictions = ensemble.predict_with_ensembles(data)
                if all_predictions and 'ensemble_models' in all_predictions:
                    ensemble_predictions = all_predictions['ensemble_models']
        except Exception as e:
            print(f"Ensemble prediction error: {e}")
        
        # Generate recommendations
        recommendations = generate_recommendations(data, base_predictions)
        
        # Prepare response
        response_data = {
            'base_models': base_predictions,
            'ensemble_models': ensemble_predictions,
            'recommendations': recommendations,
            'input_data': data
        }
        
        return JsonResponse(response_data)
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Prediction error: {str(e)}'}, status=500)

def generate_recommendations(input_data, prediction):
    """Generate personalized recommendations based on input data and prediction"""
    recommendations = []
    
    # Get average prediction score
    scores = []
    for model_result in prediction.values():
        if model_result.get('probability'):
            scores.append(model_result['probability'][1])
        else:
            scores.append(model_result.get('prediction', 0))
    
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Generate recommendations based on input data and score
    if avg_score < 0.5:
        recommendations.append("Consider enrolling in a review center to improve your chances.")
        if int(input_data.get('study_hours', 0)) < 8:
            recommendations.append("Increase your daily study hours to at least 8-10 hours.")
    
    if input_data.get('mock_exam_score', 0) < 3:
        recommendations.append("Focus on improving mock exam scores through practice tests.")
    
    if input_data.get('confidence', 0) < 3:
        recommendations.append("Work on building confidence through preparation and practice.")
    
    if not recommendations:
        recommendations.append("Keep up the good work! Continue your current study routine.")
    
    return recommendations

@require_http_methods(["GET"])
def model_status(request):
    """Get current model status"""
    try:
        from models.base_models import SocialWorkPredictorModels
        from models.ensembles import EnsembleModels
        
        predictor = SocialWorkPredictorModels()
        base_models_loaded = predictor.load_models('saved_base_models')
        
        status = {
            'base_models_trained': base_models_loaded and bool(predictor.models),
            'models': {}
        }
        
        if base_models_loaded and predictor.models:
            # Add mock performance data for loaded models
            for model_name in predictor.models.keys():
                status['models'][model_name] = {
                    'trained': True,
                    'accuracy': 0.85,  # You can store and retrieve actual metrics
                    'cv_mean': 0.82,
                    'last_trained': '2024-01-01'  # You can store actual training dates
                }
        
        return JsonResponse(status)
        
    except Exception as e:
        return JsonResponse({
            'error': f'Model status error: {str(e)}',
            'base_models_trained': False,
            'models': {}
        })

@csrf_exempt
@require_http_methods(["POST"])
def train_models_view(request):
    """Train models with uploaded CSV data"""
    try:
        if 'csv_file' not in request.FILES:
            return JsonResponse({'error': 'No CSV file uploaded'}, status=400)
        
        csv_file = request.FILES['csv_file']
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            for chunk in csv_file.chunks():
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
        
        try:
            from models.base_models import SocialWorkPredictorModels
            from models.ensembles import EnsembleModels
            
            # Initialize and train base models
            predictor = SocialWorkPredictorModels()
            df = predictor.load_data_from_csv(tmp_file_path)
            
            if df is None:
                return JsonResponse({'error': 'Failed to load CSV data'}, status=400)
            
            # Preprocess and split data
            X, y = predictor.preprocess_data(df)
            if X is None or y is None:
                return JsonResponse({'error': 'Failed to preprocess data'}, status=400)
            
            predictor.split_data(X, y)
            
            # Train base models
            base_results = predictor.train_base_models()
            predictor.save_models('saved_base_models')
            
            # Train ensemble models
            ensemble = EnsembleModels(predictor)
            ensemble_results = ensemble.train_ensembles()
            ensemble.save_ensembles('saved_ensemble_models')
            
            return JsonResponse({
                'message': 'Models trained successfully!',
                'base_results': {name: result['accuracy'] for name, result in base_results.items()},
                'ensemble_results': {name: result['accuracy'] for name, result in ensemble_results.items()}
            })
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        return JsonResponse({'error': f'Training error: {str(e)}'}, status=500)