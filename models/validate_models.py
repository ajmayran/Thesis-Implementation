import os
import joblib
import sys

def validate_pickle_file(filepath):
    """
    Validate a pickle file can be loaded
    """
    try:
        if not os.path.exists(filepath):
            return False, "File not found"
        
        file_size = os.path.getsize(filepath)
        if file_size < 100:
            return False, f"File too small ({file_size} bytes)"
        
        # Use joblib.load instead of pickle.load
        obj = joblib.load(filepath)
        
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 70)
    print("MODEL FILES VALIDATION")
    print("=" * 70)
    
    base_models_dir = 'saved_base_models'
    ensemble_models_dir = 'saved_ensemble_models'
    processed_data_dir = 'regression_processed_data'
    
    # Base models to check
    base_models = ['knn', 'decision_tree', 'random_forest', 'svr', 'ridge']
    
    # Ensemble models to check
    ensemble_models = ['bagging_random_forest', 'boosting_gradient_boost', 'stacking_ridge']
    
    # Artifacts to check
    artifacts = ['label_encoders', 'iterative_imputer', 'median_imputer', 'scaler']
    
    print("\nBase Models:")
    for model in base_models:
        filepath = os.path.join(base_models_dir, f'{model}_model.pkl')
        is_valid, message = validate_pickle_file(filepath)
        status = "OK" if is_valid else "FAIL"
        print(f"   [{status}] {model}_model.pkl - {message}")
    
    print("\nEnsemble Models:")
    for model in ensemble_models:
        filepath = os.path.join(ensemble_models_dir, f'{model}_ensemble.pkl')
        is_valid, message = validate_pickle_file(filepath)
        status = "OK" if is_valid else "FAIL"
        print(f"   [{status}] {model}_ensemble.pkl - {message}")
    
    print("\nPreprocessor:")
    preprocessor_path = os.path.join(ensemble_models_dir, 'preprocessor.pkl')
    is_valid, message = validate_pickle_file(preprocessor_path)
    status = "OK" if is_valid else "FAIL"
    print(f"   [{status}] preprocessor.pkl - {message}")
    
    print("\nArtifacts:")
    for artifact in artifacts:
        filepath = os.path.join(processed_data_dir, f'{artifact}.pkl')
        is_valid, message = validate_pickle_file(filepath)
        status = "OK" if is_valid else "FAIL"
        print(f"   [{status}] {artifact}.pkl - {message}")
    
    print("\nValidation complete")

if __name__ == '__main__':
    main()