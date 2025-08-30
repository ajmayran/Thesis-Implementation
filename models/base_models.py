import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class SocialWorkPredictorModels:
    def __init__(self):
        """Initialize the prediction models for Social Work Licensure Exam"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.ensemble_model = None
        self.feature_names = []
        
    def load_and_preprocess_data(self, csv_file_path):
        """
        Load and preprocess the CSV data
        Expected columns based on your form:
        - age, gender, gpa, major_subjects, internship_grade, scholarship
        - review_center, mock_exam_score, test_anxiety, confidence
        - study_hours, sleeping_hours, income_level, employment_status
        - employment_type, parent_occup, parent_income, result (target)
        """
        try:
            # Load data
            df = pd.read_csv(csv_file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            
            # Define categorical and numerical columns
            categorical_columns = [
                'gender', 'gpa', 'major_subjects', 'internship_grade', 'scholarship',
                'review_center', 'income_level', 'employment_status', 'employment_type',
                'parent_occup', 'parent_income'
            ]
            
            numerical_columns = [
                'age', 'mock_exam_score', 'test_anxiety', 'confidence',
                'study_hours', 'sleeping_hours'
            ]
            
            # Handle missing values
            df = df.dropna()
            
            # Encode categorical variables
            for col in categorical_columns:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
            
            # Prepare features and target
            feature_columns = categorical_columns + numerical_columns
            self.feature_names = [col for col in feature_columns if col in df.columns]
            
            X = df[self.feature_names]
            y = df['result'] if 'result' in df.columns else df['passed']  # Assuming target column
            
            # Encode target variable if it's categorical
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
                self.label_encoders['target'] = le_target
            
            # Scale numerical features
            X_scaled = X.copy()
            if numerical_columns:
                num_cols_present = [col for col in numerical_columns if col in X.columns]
                X_scaled[num_cols_present] = self.scaler.fit_transform(X[num_cols_present])
            
            return X_scaled, y
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def train_base_models(self, X, y, test_size=0.2, random_state=42):
        """Train the three base models: KNN, Decision Tree, Random Forest"""
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Initialize models
        self.models = {
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                algorithm='auto',
                metric='minkowski'
            ),
            'decision_tree': DecisionTreeClassifier(
                criterion='gini',
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                criterion='gini',
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state
            )
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name.upper()}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"{name.upper()} - Accuracy: {accuracy:.4f}")
            print(f"{name.upper()} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store test data for ensemble training
        self.X_test = X_test
        self.y_test = y_test
        
        return results
    
    def create_ensemble_model(self, X, y, n_estimators=10, random_state=42):
        """Create ensemble model using Bagging with the three base models"""
        
        print("\nCreating Ensemble Model with Bagging...")
        
        # Split data again for ensemble training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        
        # Create bagging ensembles for each base model
        ensemble_models = {
            'bagging_knn': BaggingClassifier(
                base_estimator=KNeighborsClassifier(n_neighbors=5),
                n_estimators=n_estimators,
                random_state=random_state
            ),
            'bagging_dt': BaggingClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=10, random_state=random_state),
                n_estimators=n_estimators,
                random_state=random_state
            ),
            'bagging_rf': BaggingClassifier(
                base_estimator=RandomForestClassifier(n_estimators=50, random_state=random_state),
                n_estimators=n_estimators,
                random_state=random_state
            )
        }
        
        # Train ensemble models
        ensemble_results = {}
        
        for name, model in ensemble_models.items():
            print(f"Training {name}...")
            
            # Train the ensemble model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            ensemble_results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}")
            print(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Select best ensemble model
        best_model_name = max(ensemble_results.keys(), 
                            key=lambda x: ensemble_results[x]['accuracy'])
        self.ensemble_model = ensemble_models[best_model_name]
        
        print(f"\nBest ensemble model: {best_model_name}")
        print(f"Best accuracy: {ensemble_results[best_model_name]['accuracy']:.4f}")
        
        return ensemble_results, best_model_name
    
    def predict_single(self, input_data):
        """
        Make prediction for a single instance
        input_data should be a dictionary with keys matching form fields
        """
        try:
            # Convert input to DataFrame
            df_input = pd.DataFrame([input_data])
            
            # Encode categorical variables using saved encoders
            categorical_columns = [
                'gender', 'gpa', 'major_subjects', 'internship_grade', 'scholarship',
                'review_center', 'income_level', 'employment_status', 'employment_type',
                'parent_occup', 'parent_income'
            ]
            
            for col in categorical_columns:
                if col in df_input.columns and col in self.label_encoders:
                    # Handle unseen categories
                    try:
                        df_input[col] = self.label_encoders[col].transform(df_input[col].astype(str))
                    except ValueError:
                        # If category not seen during training, use most frequent category
                        df_input[col] = 0
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df_input.columns:
                    df_input[feature] = 0
            
            # Reorder columns to match training data
            df_input = df_input[self.feature_names]
            
            # Scale numerical features
            numerical_columns = ['age', 'mock_exam_score', 'test_anxiety', 'confidence', 
                               'study_hours', 'sleeping_hours']
            num_cols_present = [col for col in numerical_columns if col in df_input.columns]
            
            if num_cols_present:
                df_input[num_cols_present] = self.scaler.transform(df_input[num_cols_present])
            
            # Make predictions using all models
            predictions = {}
            
            # Base models predictions
            for name, model in self.models.items():
                pred = model.predict(df_input)[0]
                prob = model.predict_proba(df_input)[0] if hasattr(model, 'predict_proba') else None
                predictions[name] = {
                    'prediction': int(pred),
                    'probability': prob.tolist() if prob is not None else None
                }
            
            # Ensemble model prediction
            if self.ensemble_model:
                pred = self.ensemble_model.predict(df_input)[0]
                prob = self.ensemble_model.predict_proba(df_input)[0]
                predictions['ensemble'] = {
                    'prediction': int(pred),
                    'probability': prob.tolist()
                }
            
            return predictions
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        importance_data = {}
        
        if 'decision_tree' in self.models:
            importance_data['decision_tree'] = dict(zip(
                self.feature_names, 
                self.models['decision_tree'].feature_importances_
            ))
        
        if 'random_forest' in self.models:
            importance_data['random_forest'] = dict(zip(
                self.feature_names, 
                self.models['random_forest'].feature_importances_
            ))
        
        return importance_data
    
    def save_models(self, models_dir='saved_models'):
        """Save all trained models and preprocessors"""
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Save base models
        for name, model in self.models.items():
            joblib.dump(model, f'{models_dir}/{name}_model.joblib')
        
        # Save ensemble model
        if self.ensemble_model:
            joblib.dump(self.ensemble_model, f'{models_dir}/ensemble_model.joblib')
        
        # Save preprocessors
        joblib.dump(self.label_encoders, f'{models_dir}/label_encoders.joblib')
        joblib.dump(self.scaler, f'{models_dir}/scaler.joblib')
        joblib.dump(self.feature_names, f'{models_dir}/feature_names.joblib')
        
        print(f"Models saved to {models_dir}/")
    
    def load_models(self, models_dir='saved_models'):
        """Load saved models and preprocessors"""
        try:
            # Load base models
            model_files = ['knn_model.joblib', 'decision_tree_model.joblib', 'random_forest_model.joblib']
            model_names = ['knn', 'decision_tree', 'random_forest']
            
            for name, file in zip(model_names, model_files):
                if os.path.exists(f'{models_dir}/{file}'):
                    self.models[name] = joblib.load(f'{models_dir}/{file}')
            
            # Load ensemble model
            if os.path.exists(f'{models_dir}/ensemble_model.joblib'):
                self.ensemble_model = joblib.load(f'{models_dir}/ensemble_model.joblib')
            
            # Load preprocessors
            self.label_encoders = joblib.load(f'{models_dir}/label_encoders.joblib')
            self.scaler = joblib.load(f'{models_dir}/scaler.joblib')
            self.feature_names = joblib.load(f'{models_dir}/feature_names.joblib')
            
            print(f"Models loaded from {models_dir}/")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Example usage and training script
if __name__ == "__main__":
    # Initialize the predictor
    predictor = SocialWorkPredictorModels()
    
    # Example of how to use with your CSV data
    # Replace 'your_data.csv' with your actual CSV file path
    csv_file = 'your_data.csv'  # Update this path
    
    if os.path.exists(csv_file):
        # Load and preprocess data
        X, y = predictor.load_and_preprocess_data(csv_file)
        
        if X is not None and y is not None:
            print(f"Data preprocessing completed. Features shape: {X.shape}")
            
            # Train base models
            base_results = predictor.train_base_models(X, y)
            
            # Create ensemble model
            ensemble_results, best_ensemble = predictor.create_ensemble_model(X, y)
            
            # Save models
            predictor.save_models()
            
            # Example prediction
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
                    print(f"{model_name}: {result}")
            
            # Get feature importance
            importance = predictor.get_feature_importance()
            print("\nFeature Importance:")
            for model_name, features in importance.items():
                print(f"\n{model_name}:")
                sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
                for feature, score in sorted_features[:5]:  # Top 5 features
                    print(f"  {feature}: {score:.4f}")
    
    else:
        print(f"CSV file '{csv_file}' not found. Please update the file path.")
        print("Creating sample data structure...")
        
        # Create sample DataFrame structure for reference
        sample_data = {
            'age': [25, 24, 26, 23, 27],
            'gender': ['female', 'male', 'female', 'male', 'female'],
            'gpa': ['VG', 'G', 'E', 'S', 'G'],
            'major_subjects': ['G', 'VG', 'E', 'G', 'VG'],
            'internship_grade': ['G', 'G', 'VG', 'S', 'G'],
            'scholarship': ['no', 'yes', 'no', 'no', 'yes'],
            'review_center': ['yes', 'yes', 'no', 'yes', 'yes'],
            'mock_exam_score': [4, 3, 5, 3, 4],
            'test_anxiety': [3, 4, 2, 5, 3],
            'confidence': [4, 3, 5, 2, 4],
            'study_hours': [8, 6, 10, 5, 7],
            'sleeping_hours': [7, 6, 8, 5, 7],
            'income_level': ['middle', 'low', 'high', 'low', 'middle'],
            'employment_status': ['unemployed', 'skilled', 'professional', 'unemployed', 'skilled'],
            'employment_type': ['regular', 'contractual', 'regular', 'regular', 'contractual'],
            'parent_occup': ['skilled', 'professional', 'skilled', 'unskilled', 'professional'],
            'parent_income': ['middle', 'high', 'middle', 'low', 'high'],
            'result': [1, 0, 1, 0, 1]  # 1 = Pass, 0 = Fail
        }
        
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv('sample_social_work_data.csv', index=False)
        print("Sample CSV file 'sample_social_work_data.csv' created for reference.")