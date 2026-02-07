import joblib
import json
import os
import sys

print("=" * 70)
print("REGENERATING PREPROCESSOR")
print("=" * 70)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import with experimental flag
from sklearn.experimental import enable_iterative_imputer
from regression_preprocessor import RegressionPreprocessor

# Define paths
processed_data_dir = os.path.join(current_dir, 'regression_processed_data')
ensemble_models_dir = os.path.join(current_dir, 'saved_ensemble_models')

# Ensure output directory exists
os.makedirs(ensemble_models_dir, exist_ok=True)

try:
    # Step 1: Load preprocessing components
    print("\nStep 1: Loading preprocessing components...")
    
    label_encoders_path = os.path.join(processed_data_dir, 'label_encoders.pkl')
    iterative_imputer_path = os.path.join(processed_data_dir, 'iterative_imputer.pkl')
    median_imputer_path = os.path.join(processed_data_dir, 'median_imputer.pkl')
    scaler_path = os.path.join(processed_data_dir, 'scaler.pkl')
    feature_names_path = os.path.join(processed_data_dir, 'feature_names.json')
    imputation_config_path = os.path.join(processed_data_dir, 'imputation_config.json')
    
    # Verify all files exist
    required_files = [
        label_encoders_path,
        iterative_imputer_path,
        median_imputer_path,
        scaler_path,
        feature_names_path,
        imputation_config_path
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
        print(f"   Found: {os.path.basename(file_path)}")
    
    # Load components
    print("\nStep 2: Loading components...")
    label_encoders = joblib.load(label_encoders_path)
    print(f"   Loaded label_encoders: {len(label_encoders)} encoders")
    
    iterative_imputer = joblib.load(iterative_imputer_path)
    print(f"   Loaded iterative_imputer: {type(iterative_imputer).__name__}")
    
    median_imputer = joblib.load(median_imputer_path)
    print(f"   Loaded median_imputer: {type(median_imputer).__name__}")
    
    scaler = joblib.load(scaler_path)
    print(f"   Loaded scaler: {type(scaler).__name__}")
    
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    print(f"   Loaded feature_names: {len(feature_names)} features")
    
    with open(imputation_config_path, 'r') as f:
        imputation_config = json.load(f)
    print(f"   Loaded imputation_config")
    
    # Step 3: Create preprocessor
    print("\nStep 3: Creating RegressionPreprocessor...")
    preprocessor = RegressionPreprocessor(
        iterative_imputer=iterative_imputer,
        median_imputer=median_imputer,
        label_encoders=label_encoders,
        scaler=scaler,
        imputation_config=imputation_config
    )
    print(f"   Created: {preprocessor}")
    
    # Step 4: Save preprocessor with specific protocol
    print("\nStep 4: Saving preprocessor...")
    preprocessor_path = os.path.join(ensemble_models_dir, 'preprocessor.pkl')
    
    # Remove old file if exists
    if os.path.exists(preprocessor_path):
        os.remove(preprocessor_path)
        print(f"   Removed old preprocessor file")
    
    # Save with protocol 4 for compatibility
    with open(preprocessor_path, 'wb') as f:
        joblib.dump(preprocessor, f, compress=3, protocol=4)
    
    file_size = os.path.getsize(preprocessor_path)
    print(f"   Saved: {preprocessor_path}")
    print(f"   File size: {file_size:,} bytes")
    
    # Step 5: Verify saved preprocessor
    print("\nStep 5: Verifying saved preprocessor...")
    try:
        test_load = joblib.load(preprocessor_path)
        print(f"   Successfully loaded: {test_load}")
        print(f"   Feature count: {len(test_load.get_feature_names())}")
        print(f"   Categorical columns: {len(test_load.categorical_columns)}")
        print(f"   Numerical columns: {len(test_load.numerical_columns)}")
        print(f"   Binary columns: {len(test_load.binary_columns)}")
    except Exception as e:
        print(f"   ERROR: Failed to load saved preprocessor: {e}")
        raise
    
    # Step 6: Copy additional files
    print("\nStep 6: Copying additional files...")
    
    # Copy feature names
    output_feature_names_path = os.path.join(ensemble_models_dir, 'feature_names.pkl')
    with open(output_feature_names_path, 'wb') as f:
        joblib.dump(feature_names, f, compress=3, protocol=4)
    print(f"   Saved: feature_names.pkl ({os.path.getsize(output_feature_names_path):,} bytes)")
    
    # Copy imputation config
    output_config_path = os.path.join(ensemble_models_dir, 'imputation_config.json')
    with open(output_config_path, 'w') as f:
        json.dump(imputation_config, f, indent=2)
    print(f"   Saved: imputation_config.json ({os.path.getsize(output_config_path):,} bytes)")
    
    # Step 7: Final summary
    print("\n" + "=" * 70)
    print("PREPROCESSOR REGENERATION COMPLETE")
    print("=" * 70)
    print("\nFiles created in saved_ensemble_models/:")
    print(f"   1. preprocessor.pkl ({file_size:,} bytes)")
    print(f"   2. feature_names.pkl")
    print(f"   3. imputation_config.json")
    print("\nYou can now use the prediction system.")
    
except FileNotFoundError as e:
    print("\n" + "=" * 70)
    print("ERROR: MISSING REQUIRED FILES")
    print("=" * 70)
    print(f"\n{e}")
    print("\nTo fix this, run the preprocessing notebook first:")
    print("   1. Open: models/notebook/preprocessing_notebook.ipynb")
    print("   2. Run all cells to generate required preprocessing files")
    print("   3. Then run this script again")
    sys.exit(1)
    
except Exception as e:
    print("\n" + "=" * 70)
    print("ERROR: UNEXPECTED FAILURE")
    print("=" * 70)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("   1. Verify all files in regression_processed_data/ are valid")
    print("   2. Check scikit-learn version compatibility")
    print("   3. Try running the preprocessing notebook again")
    sys.exit(1)