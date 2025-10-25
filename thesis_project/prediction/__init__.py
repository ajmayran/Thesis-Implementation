import sys
import os
import numpy as np
import joblib

# Get the parent directory (Thesis-Implementation)
parent_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..',
    '..'
))

if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

# Import from models package
from models.regression_preprocessor import RegressionPreprocessor

def load_selected_model(model_name='random_forest', model_type='base'):
    try:
        if model_type == 'base':
            models_path = os.path.join(parent_path, 'models')
            model_dir = os.path.join(models_path, 'saved_base_models')
            model_file = f'{model_name}_model.pkl'
        else:
            models_path = os.path.join(parent_path, 'models')
            model_dir = os.path.join(models_path, 'saved_ensemble_models')
            model_file = f'{model_name}_ensemble.pkl'
        
        model_path = os.path.join(model_dir, model_file)
        preprocessor_path = os.path.join(
            models_path, 'saved_base_models', 'preprocessor.pkl'
        )
        
        if not os.path.exists(model_path):
            return None, None, f"Model file not found: {model_path}"
        
        if not os.path.exists(preprocessor_path):
            return None, None, f"Preprocessor file not found: {preprocessor_path}"
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        print(f"Loaded {model_type} model: {model_name}")
        return model, preprocessor, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Error loading model: {str(e)}"

# ...existing code...