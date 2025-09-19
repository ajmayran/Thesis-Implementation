"""
Script to train the Social Work Licensure Exam prediction models
Run this script to train your models with your CSV data
"""

import os
import sys
import pandas as pd
from base_models import SocialWorkPredictorModels

def main():
    """Main training function"""
    print("Social Work Licensure Exam Predictor - Model Training")
    print("=" * 55)
    
    # Initialize predictor
    predictor = SocialWorkPredictorModels()
    
    # Ask for CSV file path
    csv_file = input("Enter the path to your CSV file (or press Enter to use sample data): ").strip()
    
    if not csv_file:
        # Use sample data
        print("Creating sample dataset...")
        from base_models import create_sample_dataset
        df = create_sample_dataset(250, 'social_work_sample_250')
        print("Using generated sample data...")
    else:
        if not os.path.exists(csv_file):
            print(f"Error: File '{csv_file}' not found.")
            return
        
        print(f"Loading data from {csv_file}...")
        
        # Load data based on file extension
        if csv_file.endswith('.csv'):
            df = predictor.load_data_from_csv(csv_file)
        elif csv_file.endswith('.xlsx') or csv_file.endswith('.xls'):
            df = predictor.load_data_from_excel(csv_file)
        elif csv_file.endswith('.json'):
            df = predictor.load_data_from_json(csv_file)
        else:
            print("Unsupported file format. Please use CSV, Excel, or JSON.")
            return
    
    if df is None:
        print("Failed to load data.")
        return
    
    # Preprocess data
    print("Preprocessing data...")
    X, y = predictor.preprocess_data(df)
    
    if X is None or y is None:
        print("Failed to preprocess data.")
        return
    
    print(f"Data processed successfully. Shape: {X.shape}")
    print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Split data
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = predictor.split_data(X, y, test_size=0.2)
    
    # Train base models
    print("\nTraining base models (KNN, Decision Tree, Random Forest)...")
    base_results = predictor.train_base_models()
    
    print("\nBase Model Results:")
    print("-" * 40)
    for name, results in base_results.items():
        print(f"{name.upper()}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  CV Score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
        print()
    
    # Get feature importance
    print("Feature Importance Analysis:")
    print("-" * 40)
    importance = predictor.get_feature_importance()
    for model_name, features in importance.items():
        print(f"\n{model_name.upper()}:")
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_features[:8]:  # Top 8 features
            print(f"  {feature}: {score:.4f}")
    
    # Save models
    models_dir = input("\nEnter directory to save models (default: 'saved_models'): ").strip()
    if not models_dir:
        models_dir = 'saved_models'
    
    predictor.save_models(models_dir)
    print(f"Base models saved to {models_dir}/")
    
    # Test prediction
    print("\nTesting prediction with sample data...")
    sample_input = {
        'age': 25,
        'gender': 'female',
        'gpa': 'VG',
        'major_subjects': 'G',
        'internship_grade': 'G',
        'scholarship': 'no',
        'review_center': 'yes',
        'mock_exam_score': 4,
        'test_anxiety': 3,
        'confidence': 4,
        'study_hours': 8,
        'sleeping_hours': 7,
        'income_level': 'middle',
        'employment_status': 'unemployed',
        'employment_type': 'regular',
        'parent_occup': 'skilled',
        'parent_income': 'middle'
    }
    
    predictions = predictor.predict_single(sample_input)
    if predictions:
        print("\nSample Prediction Results:")
        for model_name, result in predictions.items():
            if result['probability']:
                pass_prob = result['probability'][1] * 100
                print(f"{model_name.upper()}: {pass_prob:.1f}% chance of passing")
            else:
                print(f"{model_name.upper()}: {'Pass' if result['prediction'] else 'Fail'}")
    
    print("\nBase model training completed successfully!")
    print("Next step: Run ensemble training using ensembles.py")

if __name__ == "__main__":
    main()