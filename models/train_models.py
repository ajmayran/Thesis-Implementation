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
    csv_file = input("Enter the path to your CSV file: ").strip()
    
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        return
    
    print(f"Loading data from {csv_file}...")
    
    # Load and preprocess data
    X, y = predictor.load_and_preprocess_data(csv_file)
    
    if X is None or y is None:
        print("Failed to load and preprocess data.")
        return
    
    print(f"Data loaded successfully. Shape: {X.shape}")
    print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Train base models
    print("\nTraining base models...")
    base_results = predictor.train_base_models(X, y)
    
    print("\nBase Model Results:")
    print("-" * 40)
    for name, results in base_results.items():
        print(f"{name.upper()}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  CV Score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
        print()
    
    # Create ensemble model
    print("Creating ensemble models...")
    ensemble_results, best_ensemble = predictor.create_ensemble_model(X, y)
    
    print("\nEnsemble Model Results:")
    print("-" * 40)
    for name, results in ensemble_results.items():
        print(f"{name.upper()}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  CV Score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
        print()
    
    print(f"Best ensemble model: {best_ensemble}")
    
    # Get feature importance
    print("\nFeature Importance:")
    print("-" * 40)
    importance = predictor.get_feature_importance()
    for model_name, features in importance.items():
        print(f"\n{model_name.upper()}:")
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_features:
            print(f"  {feature}: {score:.4f}")
    
    # Save models
    models_dir = input("\nEnter directory to save models (default: 'saved_models'): ").strip()
    if not models_dir:
        models_dir = 'saved_models'
    
    predictor.save_models(models_dir)
    print(f"Models saved to {models_dir}/")
    
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
                print(f"{model_name}: {pass_prob:.1f}% chance of passing")
            else:
                print(f"{model_name}: {'Pass' if result['prediction'] else 'Fail'}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()