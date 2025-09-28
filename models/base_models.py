import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os

class SocialWorkPredictorModels:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_names = []
        
    def load_data_from_csv(self, file_path):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded from CSV successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
    
    def preprocess_data(self, df):
        """
        Preprocess the loaded data using OneHotEncoder for categorical variables
        """
        try:
            # Make a copy to avoid modifying original data
            df = df.copy()
            
            # Handle missing values
            df = df.dropna()
            print(f"After removing missing values. Shape: {df.shape}")
            
            # Define categorical and numerical columns based on your CSV
            categorical_columns = ['Gender', 'IncomeLevel', 'EmploymentStatus']
            numerical_columns = ['Age', 'StudyHours', 'SleepHours', 'Confidence', 
                               'MockExamScore', 'GPA', 'Scholarship', 'InternshipGrade', 
                               'ExamResultPercent']
            
            # Handle ReviewCenter separately (it's binary: 0/1)
            binary_columns = ['ReviewCenter']
            
            # Prepare features (X) and target (y)
            all_feature_columns = categorical_columns + numerical_columns + binary_columns
            X = df[all_feature_columns].copy()
            y = df['Passed'].values
            
            # Create preprocessor using ColumnTransformer
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_columns + binary_columns),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
                ],
                remainder='passthrough'
            )
            
            # Fit and transform the data
            X_processed = self.preprocessor.fit_transform(X)
            
            # Get feature names for the transformed data
            # Numerical feature names
            num_feature_names = numerical_columns + binary_columns
            
            # Categorical feature names (after one-hot encoding)
            cat_feature_names = []
            for i, col in enumerate(categorical_columns):
                # Get categories for this column (excluding the first one due to drop='first')
                categories = self.preprocessor.named_transformers_['cat'].categories_[i][1:]
                cat_feature_names.extend([f"{col}_{cat}" for cat in categories])
            
            self.feature_names = num_feature_names + cat_feature_names
            
            print(f"Features processed. Shape: {X_processed.shape}")
            print(f"Feature names: {self.feature_names}")
            print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
            
            return X_processed, y
            
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return None, None
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def train_base_models(self):
        """Train base models with hyperparameter tuning"""
        if not hasattr(self, 'X_train') or self.X_train is None:
            print("Please load and preprocess data first")
            return None
            
        results = {}
        
        # Define models with hyperparameters for tuning
        model_configs = {
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
            }
        }
        
        for name, config in model_configs.items():
            print(f"\nTraining {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Best model
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            cv_scores = cross_val_score(best_model, self.X_train, self.y_train, cv=5)
            
            # Store model and results
            self.models[name] = best_model
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'best_params': grid_search.best_params_,
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Best parameters: {grid_search.best_params_}")
        
        return results
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            return None
            
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            # Sort by importance
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            return None
    
    def predict_single(self, input_data):
        """Make prediction for a single input"""
        try:
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                # Map input keys to match CSV column names
                column_mapping = {
                    'age': 'Age',
                    'gender': 'Gender', 
                    'study_hours': 'StudyHours',
                    'sleep_hours': 'SleepHours',
                    'review_center': 'ReviewCenter',
                    'confidence': 'Confidence',
                    'mock_exam_score': 'MockExamScore',
                    'gpa': 'GPA',
                    'scholarship': 'Scholarship',
                    'internship_grade': 'InternshipGrade',
                    'income_level': 'IncomeLevel',
                    'employment_status': 'EmploymentStatus',
                    'exam_result_percent': 'ExamResultPercent'
                }
                
                mapped_data = {}
                for key, value in input_data.items():
                    if key in column_mapping:
                        mapped_data[column_mapping[key]] = value
                    else:
                        mapped_data[key] = value
                
                df_input = pd.DataFrame([mapped_data])
            else:
                df_input = pd.DataFrame([input_data])
            
            # Prepare features in the same order as training
            categorical_columns = ['Gender', 'IncomeLevel', 'EmploymentStatus']
            numerical_columns = ['Age', 'StudyHours', 'SleepHours', 'Confidence', 
                               'MockExamScore', 'GPA', 'Scholarship', 'InternshipGrade', 
                               'ExamResultPercent']
            binary_columns = ['ReviewCenter']
            
            all_feature_columns = categorical_columns + numerical_columns + binary_columns
            X_input = df_input[all_feature_columns].copy()
            
            # Apply same preprocessing using the fitted preprocessor
            X_input_processed = self.preprocessor.transform(X_input)
            
            # Make predictions with all models
            predictions = {}
            for name, model in self.models.items():
                pred = model.predict(X_input_processed)[0]
                pred_proba = model.predict_proba(X_input_processed)[0] if hasattr(model, 'predict_proba') else None
                
                predictions[name] = {
                    'prediction': int(pred),
                    'probability': pred_proba.tolist() if pred_proba is not None else None
                }
            
            return predictions
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def save_models(self, directory='saved_models'):
        """Save trained models and preprocessor"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(directory, f'{name}_model.pkl'))
        
        # Save preprocessor (contains OneHotEncoder and StandardScaler)
        joblib.dump(self.preprocessor, os.path.join(directory, 'preprocessor.pkl'))
        
        # Save feature names
        joblib.dump(self.feature_names, os.path.join(directory, 'feature_names.pkl'))
        
        print(f"Models saved to {directory}")
    
    def load_models(self, directory='saved_models'):
        """Load trained models and preprocessor"""
        try:
            # Load models
            model_files = ['knn_model.pkl', 'decision_tree_model.pkl', 'random_forest_model.pkl',
                          'logistic_regression_model.pkl', 'svm_model.pkl', 'naive_bayes_model.pkl']
            
            for file in model_files:
                model_path = os.path.join(directory, file)
                if os.path.exists(model_path):
                    model_name = file.replace('_model.pkl', '')
                    self.models[model_name] = joblib.load(model_path)
            
            # Load preprocessor
            self.preprocessor = joblib.load(os.path.join(directory, 'preprocessor.pkl'))
            self.feature_names = joblib.load(os.path.join(directory, 'feature_names.pkl'))
            
            print(f"Models loaded from {directory}")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

def main():
    """Main training function using your existing CSV"""
    predictor = SocialWorkPredictorModels()
    
    # Load your existing CSV file
    csv_file = 'social_work_exam_dataset.csv'
    
    print("Loading data...")
    df = predictor.load_data_from_csv(csv_file)
    
    if df is None:
        print("Failed to load data.")
        return
    
    # Display basic info about the dataset
    print(f"\nDataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nCategorical columns unique values:")
    categorical_cols = ['Gender', 'IncomeLevel', 'EmploymentStatus']
    for col in categorical_cols:
        print(f"{col}: {df[col].unique()}")
    
    print(f"\nTarget distribution:")
    print(df['Passed'].value_counts())
    print(f"Pass rate: {df['Passed'].mean():.2%}")
    
    # Preprocess data with OneHotEncoder
    print("\nPreprocessing data with OneHotEncoder...")
    X, y = predictor.preprocess_data(df)
    
    if X is None or y is None:
        print("Failed to preprocess data.")
        return
    
    print(f"Data processed successfully. Shape: {X.shape}")
    
    # Split data
    print("\nSplitting data into train/test sets...")
    predictor.X_train, predictor.X_test, predictor.y_train, predictor.y_test = predictor.split_data(X, y, test_size=0.2)
    
    # Train base models
    print("\nTraining models with hyperparameter tuning...")
    results = predictor.train_base_models()
    
    if results:
        print("\n" + "="*50)
        print("TRAINING RESULTS SUMMARY")
        print("="*50)
        
        # Sort results by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for model_name, metrics in sorted_results:
            print(f"\n{model_name.upper()}:")
            print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"  CV Mean: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")
            print(f"  Best Parameters: {metrics['best_params']}")
        
        # Show feature importance
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE (Random Forest)")
        print("="*50)
        
        feature_importance = predictor.get_feature_importance('random_forest')
        if feature_importance:
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:15], 1):
                print(f"{i:2d}. {feature:<30}: {importance:.4f}")
        
        # Save models
        print("\nSaving models...")
        predictor.save_models()
        
        # Example prediction
        print("\n" + "="*50)
        print("EXAMPLE PREDICTION")
        print("="*50)
        
        # Use data from your CSV for example
        sample_data = {
            'age': 25,
            'gender': 'Female',
            'study_hours': 6,
            'sleep_hours': 7,
            'review_center': 1,
            'confidence': 4,
            'mock_exam_score': 85,
            'gpa': 1.5,
            'scholarship': 1,
            'internship_grade': 90,
            'income_level': 'Middle',
            'employment_status': 'Unemployed',
            'exam_result_percent': 85
        }
        
        prediction = predictor.predict_single(sample_data)
        
        if prediction:
            print(f"Input: {sample_data}")
            print("\nPredictions:")
            for model_name, result in prediction.items():
                prob_text = ""
                if result['probability']:
                    prob_text = f" (Probability: {result['probability'][1]:.3f})"
                pass_fail = "PASS" if result['prediction'] == 1 else "FAIL"
                print(f"  {model_name:20s}: {pass_fail}{prob_text}")
        
        print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()