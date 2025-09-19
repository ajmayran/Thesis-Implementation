import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json

class SocialWorkPredictorModels:
    def __init__(self):
        """Initialize the prediction models for Social Work Licensure Exam"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data_from_excel(self, file_path):
        """Load data from Excel file"""
        try:
            df = pd.read_excel(file_path)
            print(f"Data loaded from Excel successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return None
    
    def load_data_from_json(self, file_path):
        """Load data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            print(f"Data loaded from JSON successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None
    
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
        Preprocess the loaded data
        Expected columns:
        - age, gender, gpa, major_subjects, internship_grade, scholarship
        - review_center, mock_exam_score, test_anxiety, confidence
        - study_hours, sleeping_hours, income_level, employment_status
        - employment_type, parent_occup, parent_income, result (target)
        """
        try:
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
            print(f"After removing missing values. Shape: {df.shape}")
            
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
            y = df['result'] if 'result' in df.columns else df['passed']
            
            # Encode target variable if it's categorical
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
                self.label_encoders['target'] = le_target
            
            # Scale numerical features
            X_scaled = X.copy()
            if numerical_columns:
                num_cols_present = [col for col in numerical_columns if col in X.columns]
                if num_cols_present:
                    X_scaled[num_cols_present] = self.scaler.fit_transform(X[num_cols_present])
            
            return X_scaled, y
            
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return None, None
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Training target distribution: {pd.Series(self.y_train).value_counts().to_dict()}")
        print(f"Test target distribution: {pd.Series(self.y_test).value_counts().to_dict()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_base_models(self, random_state=42):
        """Train the three base models: KNN, Decision Tree, Random Forest"""
        
        if self.X_train is None or self.y_train is None:
            print("No training data available. Please split data first.")
            return None
        
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
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(self.y_test, y_pred),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            print(f"{name.upper()} - Accuracy: {accuracy:.4f}")
            print(f"{name.upper()} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def predict_single(self, input_data):
        """
        Make prediction for a single instance using all three base models
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
            
            # Make predictions using all three base models
            predictions = {}
            
            for name, model in self.models.items():
                pred = model.predict(df_input)[0]
                prob = model.predict_proba(df_input)[0] if hasattr(model, 'predict_proba') else None
                predictions[name] = {
                    'prediction': int(pred),
                    'probability': prob.tolist() if prob is not None else None
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
        
        # Save preprocessors
        joblib.dump(self.label_encoders, f'{models_dir}/label_encoders.joblib')
        joblib.dump(self.scaler, f'{models_dir}/scaler.joblib')
        joblib.dump(self.feature_names, f'{models_dir}/feature_names.joblib')
        
        print(f"Base models saved to {models_dir}/")
    
    def load_models(self, models_dir='saved_models'):
        """Load saved models and preprocessors"""
        try:
            # Load base models
            model_files = ['knn_model.joblib', 'decision_tree_model.joblib', 'random_forest_model.joblib']
            model_names = ['knn', 'decision_tree', 'random_forest']
            
            for name, file in zip(model_names, model_files):
                if os.path.exists(f'{models_dir}/{file}'):
                    self.models[name] = joblib.load(f'{models_dir}/{file}')
            
            # Load preprocessors
            if os.path.exists(f'{models_dir}/label_encoders.joblib'):
                self.label_encoders = joblib.load(f'{models_dir}/label_encoders.joblib')
            if os.path.exists(f'{models_dir}/scaler.joblib'):
                self.scaler = joblib.load(f'{models_dir}/scaler.joblib')
            if os.path.exists(f'{models_dir}/feature_names.joblib'):
                self.feature_names = joblib.load(f'{models_dir}/feature_names.joblib')
            
            print(f"Base models loaded from {models_dir}/")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

def create_sample_dataset(n_samples=250, filename='sample_social_work_data'):
    """Create sample dataset for training with 250 samples"""
    
    np.random.seed(42)  # For reproducible results
    
    # Define possible values for categorical variables
    genders = ['male', 'female']
    gpas = ['E', 'VG', 'G', 'S', 'F']  # Excellent, Very Good, Good, Satisfactory, Fail
    major_subjects = ['E', 'VG', 'G', 'S', 'F']
    internship_grades = ['E', 'VG', 'G', 'S', 'F']
    scholarships = ['yes', 'no']
    review_centers = ['yes', 'no']
    income_levels = ['low', 'middle', 'high']
    employment_statuses = ['unemployed', 'skilled', 'professional']
    employment_types = ['regular', 'contractual', 'part-time']
    parent_occupations = ['unskilled', 'skilled', 'professional']
    parent_incomes = ['low', 'middle', 'high']
    
    # Generate sample data
    data = []
    
    for i in range(n_samples):
        # Generate realistic correlations
        gpa = np.random.choice(gpas, p=[0.1, 0.25, 0.35, 0.25, 0.05])  # More likely to have good grades
        
        # Age between 21-30, most likely around 23-25
        age = np.random.normal(24, 2)
        age = max(21, min(30, int(age)))
        
        # Mock exam score (1-5) - correlated with GPA
        gpa_to_score = {'E': [4, 5], 'VG': [3, 4, 5], 'G': [2, 3, 4], 'S': [1, 2, 3], 'F': [1, 2]}
        mock_exam_score = np.random.choice(gpa_to_score[gpa])
        
        # Study hours (4-16) - higher for better students
        if gpa in ['E', 'VG']:
            study_hours = np.random.randint(8, 16)
        elif gpa == 'G':
            study_hours = np.random.randint(6, 12)
        else:
            study_hours = np.random.randint(4, 10)
        
        # Sleeping hours (5-9)
        sleeping_hours = np.random.randint(5, 10)
        
        # Test anxiety (1-5) - inversely correlated with confidence
        test_anxiety = np.random.randint(1, 6)
        
        # Confidence (1-5) - somewhat inversely correlated with anxiety
        if test_anxiety <= 2:
            confidence = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])
        elif test_anxiety >= 4:
            confidence = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        else:
            confidence = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
        
        # Review center attendance - better students more likely to attend
        if gpa in ['E', 'VG']:
            review_center = np.random.choice(review_centers, p=[0.8, 0.2])
        else:
            review_center = np.random.choice(review_centers, p=[0.6, 0.4])
        
        # Result - higher probability of passing for better students
        pass_probability = 0.2  # Base probability
        
        # Adjust based on GPA
        if gpa == 'E': pass_probability += 0.6
        elif gpa == 'VG': pass_probability += 0.4
        elif gpa == 'G': pass_probability += 0.2
        elif gpa == 'S': pass_probability += 0.1
        
        # Adjust based on mock exam score
        pass_probability += (mock_exam_score - 1) * 0.1
        
        # Adjust based on study hours
        pass_probability += (study_hours - 8) * 0.02
        
        # Adjust based on review center
        if review_center == 'yes':
            pass_probability += 0.15
        
        # Adjust based on confidence and anxiety
        pass_probability += (confidence - 3) * 0.05
        pass_probability -= (test_anxiety - 3) * 0.03
        
        # Ensure probability is between 0 and 1
        pass_probability = max(0, min(1, pass_probability))
        
        result = 1 if np.random.random() < pass_probability else 0
        
        # Create record
        record = {
            'age': age,
            'gender': np.random.choice(genders),
            'gpa': gpa,
            'major_subjects': np.random.choice(major_subjects),
            'internship_grade': np.random.choice(internship_grades),
            'scholarship': np.random.choice(scholarships, p=[0.3, 0.7]),
            'review_center': review_center,
            'mock_exam_score': mock_exam_score,
            'test_anxiety': test_anxiety,
            'confidence': confidence,
            'study_hours': study_hours,
            'sleeping_hours': sleeping_hours,
            'income_level': np.random.choice(income_levels, p=[0.3, 0.5, 0.2]),
            'employment_status': np.random.choice(employment_statuses, p=[0.6, 0.3, 0.1]),
            'employment_type': np.random.choice(employment_types, p=[0.6, 0.3, 0.1]),
            'parent_occup': np.random.choice(parent_occupations, p=[0.2, 0.5, 0.3]),
            'parent_income': np.random.choice(parent_incomes, p=[0.3, 0.5, 0.2]),
            'result': result
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save as different formats
    df.to_csv(f'{filename}.csv', index=False)
    df.to_excel(f'{filename}.xlsx', index=False)
    df.to_json(f'{filename}.json', orient='records', indent=2)
    
    print(f"Sample dataset created with {n_samples} records:")
    print(f"- CSV: {filename}.csv")
    print(f"- Excel: {filename}.xlsx") 
    print(f"- JSON: {filename}.json")
    print(f"\nTarget distribution: {df['result'].value_counts().to_dict()}")
    print(f"Pass rate: {df['result'].mean():.2%}")
    
    return df

# Example usage
if __name__ == "__main__":
    # Create sample dataset
    print("Creating sample dataset...")
    df = create_sample_dataset(250, 'social_work_sample_250')
    
    # Initialize predictor
    predictor = SocialWorkPredictorModels()
    
    # Load data (you can use CSV, Excel, or JSON)
    # df = predictor.load_data_from_csv('social_work_sample_250.csv')
    # df = predictor.load_data_from_excel('social_work_sample_250.xlsx')
    # df = predictor.load_data_from_json('social_work_sample_250.json')
    
    # Preprocess data
    X, y = predictor.preprocess_data(df)
    
    if X is not None and y is not None:
        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = predictor.split_data(X, y, test_size=0.2)
        
        # Train base models
        results = predictor.train_base_models()
        
        print("\n" + "="*50)
        print("BASE MODEL COMPARISON RESULTS")
        print("="*50)
        
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})")
        
        # Get feature importance
        importance = predictor.get_feature_importance()
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE")
        print("="*50)
        
        for model_name, features in importance.items():
            print(f"\n{model_name.upper()}:")
            sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
            for feature, score in sorted_features[:8]:  # Top 8 features
                print(f"  {feature}: {score:.4f}")
        
        # Test prediction
        print("\n" + "="*50)
        print("SAMPLE PREDICTION TEST")
        print("="*50)
        
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
            print("\nThree Model Predictions:")
            for model_name, result in predictions.items():
                if result['probability']:
                    pass_prob = result['probability'][1] * 100
                    print(f"{model_name.upper()}: {pass_prob:.1f}% chance of passing")
                else:
                    print(f"{model_name.upper()}: {'Pass' if result['prediction'] else 'Fail'}")
        
        # Save models
        predictor.save_models('saved_base_models')
        print(f"\nModels saved successfully!")