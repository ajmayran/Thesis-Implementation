import os
import sys

# Path to models directory
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')

# Active model configuration - change these to switch models
SELECTED_MODEL_NAME = 'stacking_neural_ridge_neural_final'
MODEL_CATEGORY = 'ensemble'

# Model paths
BASE_MODELS_DIR = os.path.join(models_path, 'saved_base_models')
ENSEMBLE_MODELS_DIR = os.path.join(models_path, 'saved_ensemble_models')
PREPROCESSOR_PATH = os.path.join(models_path, 'regression_processed_data', 'preprocessor.pkl')