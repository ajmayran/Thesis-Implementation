import joblib
import os

preprocessor_path = 'saved_ensemble_models/preprocessor.pkl'

print(f"Checking: {preprocessor_path}")
print(f"File exists: {os.path.exists(preprocessor_path)}")

if os.path.exists(preprocessor_path):
    print(f"File size: {os.path.getsize(preprocessor_path):,} bytes")
    
    try:
        with open(preprocessor_path, 'rb') as f:
            # Read first few bytes
            first_bytes = f.read(10)
            print(f"First bytes (hex): {first_bytes.hex()}")
            print(f"First bytes (ascii): {first_bytes}")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    try:
        obj = joblib.load(preprocessor_path)
        print(f"Successfully loaded: {type(obj)}")
        print(f"Attributes: {dir(obj)}")
    except Exception as e:
        print(f"Error loading with joblib: {e}")