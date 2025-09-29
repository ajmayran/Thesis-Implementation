import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class SocialWorkPredictorModels:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_names = []
        
        # CORRECTED: Features available BEFORE taking the exam
        self.categorical_columns = ['Gender', 'IncomeLevel', 'EmploymentStatus']
        self.numerical_columns = ['Age', 'StudyHours', 'SleepHours', 'Confidence', 
                                'MockExamScore', 'GPA', 'Scholarship', 'InternshipGrade']
        self.binary_columns = ['ReviewCenter']
        
        self.target_column = 'Passed'
        
    def load_data_from_csv(self, file_path):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            print(f"[SUCCESS] Data loaded from CSV successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"[ERROR] Error loading CSV file: {e}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess data using features available BEFORE exam"""
        try:
            df = df.copy()
            df = df.dropna()
            print(f"[CLEAN] After removing missing values. Shape: {df.shape}")
            
            # Use only PRE-EXAM features
            all_feature_columns = self.categorical_columns + self.numerical_columns + self.binary_columns
            X = df[all_feature_columns].copy()
            y = df[self.target_column].values
            
            print(f"[FEATURES] Using {len(all_feature_columns)} pre-exam features:")
            for col in all_feature_columns:
                print(f"   - {col}")
            
            # Create preprocessor
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.numerical_columns + self.binary_columns),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_columns)
                ],
                remainder='passthrough'
            )
            
            X_processed = self.preprocessor.fit_transform(X)
            
            # Get feature names
            num_feature_names = self.numerical_columns + self.binary_columns
            cat_feature_names = []
            for i, col in enumerate(self.categorical_columns):
                categories = self.preprocessor.named_transformers_['cat'].categories_[i][1:]
                cat_feature_names.extend([f"{col}_{cat}" for cat in categories])
            
            self.feature_names = num_feature_names + cat_feature_names
            
            print(f"[SUCCESS] Features processed. Final shape: {X_processed.shape}")
            return X_processed, y
            
        except Exception as e:
            print(f"[ERROR] Error preprocessing data: {e}")
            return None, None
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def train_base_models(self):
        """Train base models with hyperparameter tuning"""
        if not hasattr(self, 'X_train') or self.X_train is None:
            print("[ERROR] Please load and preprocess data first")
            return None
            
        print(f"\n[TRAINING] Training base models with {self.X_train.shape[1]} features...")
        results = {}
        
        # Model configurations
        model_configs = {
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, 15]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                }
            }
        }
        
        for name, config in model_configs.items():
            print(f"   Training {name}...")
            
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            
            y_pred = best_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            cv_scores = cross_val_score(best_model, self.X_train, self.y_train, cv=5)
            
            self.models[name] = best_model
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'best_params': grid_search.best_params_,
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            print(f"      {name}: {accuracy:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def predict_single(self, input_data):
        """Make prediction for a single candidate BEFORE they take the exam"""
        try:
            # Map input data to correct feature names
            if isinstance(input_data, dict):
                mapped_data = {}
                
                # Map common variations to standard column names
                field_mapping = {
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
                    'employment_status': 'EmploymentStatus'
                }
                
                for key, value in input_data.items():
                    mapped_key = field_mapping.get(key, key)
                    mapped_data[mapped_key] = value
                
                df_input = pd.DataFrame([mapped_data])
            else:
                df_input = pd.DataFrame([input_data])
            
            # Use only PRE-EXAM features
            all_feature_columns = self.categorical_columns + self.numerical_columns + self.binary_columns
            X_input = df_input[all_feature_columns].copy()
            
            # Apply preprocessing
            X_input_processed = self.preprocessor.transform(X_input)
            
            # Make predictions
            predictions = {}
            for name, model in self.models.items():
                pred = model.predict(X_input_processed)[0]
                pred_proba = model.predict_proba(X_input_processed)[0] if hasattr(model, 'predict_proba') else None
                
                predictions[name] = {
                    'prediction': int(pred),
                    'probability': pred_proba.tolist() if pred_proba is not None else None,
                    'pass_probability': float(pred_proba[1]) if pred_proba is not None else float(pred)
                }
            
            return predictions
            
        except Exception as e:
            print(f"[ERROR] Error making prediction: {e}")
            return None
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            return None
            
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            return None
    
    def save_models(self, directory='saved_models'):
        """Save trained models and preprocessor"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(directory, f'{name}_model.pkl'))
        
        joblib.dump(self.preprocessor, os.path.join(directory, 'preprocessor.pkl'))
        joblib.dump(self.feature_names, os.path.join(directory, 'feature_names.pkl'))
        
        print(f"[SAVE] Models saved to {directory}/")
    
    def load_models(self, directory='saved_models'):
        """Load trained models and preprocessor"""
        try:
            model_files = [f for f in os.listdir(directory) if f.endswith('_model.pkl')]
            
            for model_file in model_files:
                model_name = model_file.replace('_model.pkl', '')
                self.models[model_name] = joblib.load(os.path.join(directory, model_file))
            
            self.preprocessor = joblib.load(os.path.join(directory, 'preprocessor.pkl'))
            self.feature_names = joblib.load(os.path.join(directory, 'feature_names.pkl'))
            
            print(f"[LOAD] Loaded {len(self.models)} models from {directory}/")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading models: {e}")
            return False

def main():
    """Main training function using correct pre-exam features"""
    predictor = SocialWorkPredictorModels()
    
    # Load data
    csv_file = 'social_work_exam_dataset.csv'
    print("="*60)
    print("[START] TRAINING SOCIAL WORK EXAM PREDICTOR")
    print("="*60)
    
    df = predictor.load_data_from_csv(csv_file)
    if df is None:
        return
    
    # Show what we're actually predicting
    print(f"\n[PREDICTION GOAL]")
    print(f"   INPUT: Pre-exam factors (study habits, background, etc.)")
    print(f"   OUTPUT: Will the candidate PASS or FAIL the exam?")
    print(f"   NOTE: ExamResultPercent is excluded (that's the actual exam score!)")
    
    # Display dataset info
    print(f"\n[DATASET INFO]")
    print(f"   Shape: {df.shape}")
    print(f"   Pass Rate: {df['Passed'].mean():.2%}")
    print(f"   Average Exam Score: {df['ExamResultPercent'].mean():.1f}%")
    
    # Preprocess data
    X, y = predictor.preprocess_data(df)
    if X is None or y is None:
        return
    
    # Split data
    predictor.X_train, predictor.X_test, predictor.y_train, predictor.y_test = predictor.split_data(X, y)
    
    # Train models
    results = predictor.train_base_models()
    
    if results:
        print(f"\n[RESULTS] Model Performance:")
        print("-" * 50)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for model_name, metrics in sorted_results:
            print(f"{model_name:20s}: {metrics['accuracy']:.4f} (+/- {metrics['cv_std']*2:.4f})")
        
        # Feature importance
        feature_importance = predictor.get_feature_importance('random_forest')
        if feature_importance:
            print(f"\n[FEATURE IMPORTANCE] Top 10 Predictive Factors:")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
                print(f"{i:2d}. {feature:<20s}: {importance:.4f}")
        
        # Save models
        predictor.save_models()
        
        # Example prediction for new candidate
        print(f"\n[EXAMPLE] Predicting for a new candidate:")
        new_candidate = {
            'age': 25,
            'gender': 'Female',
            'study_hours': 8,
            'sleep_hours': 7,
            'review_center': 1,
            'confidence': 4,
            'mock_exam_score': 85,
            'gpa': 1.5,
            'scholarship': 1,
            'internship_grade': 88,
            'income_level': 'Middle',
            'employment_status': 'Unemployed'
        }
        
        predictions = predictor.predict_single(new_candidate)
        if predictions:
            print(f"Input: {new_candidate}")
            print(f"\nPredictions:")
            for model_name, result in predictions.items():
                pass_prob = result['pass_probability']
                status = "LIKELY TO PASS" if pass_prob > 0.5 else "AT RISK"
                print(f"  {model_name:20s}: {pass_prob:.1%} - {status}")
        
        print(f"\n[COMPLETE] Training completed successfully!")

if __name__ == "__main__":
    main()