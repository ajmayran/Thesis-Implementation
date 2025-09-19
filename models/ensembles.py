import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from base_models import SocialWorkPredictorModels

class EnsembleModels:
    def __init__(self, base_predictor):
        """Initialize ensemble models using trained base models"""
        self.base_predictor = base_predictor
        self.ensemble_models = {}
        
    def create_voting_ensemble(self):
        """Create soft voting ensemble from base models"""
        if not self.base_predictor.models:
            print("Base models not trained. Please train base models first.")
            return None
            
        voting_clf = VotingClassifier(
            estimators=[
                ('knn', self.base_predictor.models['knn']),
                ('dt', self.base_predictor.models['decision_tree']),
                ('rf', self.base_predictor.models['random_forest'])
            ],
            voting='soft'  # Use probability-based voting
        )
        
        self.ensemble_models['voting'] = voting_clf
        return voting_clf
    
    def create_bagging_ensemble(self, n_estimators=50, random_state=42):
        """Create bagging ensemble using Random Forest (which is already bagging)"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Random Forest is essentially bagging with decision trees
        bagging_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            bootstrap=True  # This makes it true bagging
        )
        
        self.ensemble_models['bagging'] = bagging_clf
        return bagging_clf
    
    def create_boosting_ensemble(self, n_estimators=100, random_state=42):
        """Create Gradient Boosting ensemble"""
        boosting_clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state
        )
        
        self.ensemble_models['gradient_boosting'] = boosting_clf
        return boosting_clf
    
    def create_stacking_ensemble(self, random_state=42):
        """Create stacking ensemble with Logistic Regression as meta-learner"""
        if not self.base_predictor.models:
            print("Base models not trained. Please train base models first.")
            return None
        
        # Create meta-features using cross-validation
        def create_meta_features(X, y):
            meta_features = np.zeros((X.shape[0], len(self.base_predictor.models)))
            
            for i, (name, model) in enumerate(self.base_predictor.models.items()):
                # Use cross-validation to create out-of-fold predictions
                cv_predictions = cross_val_predict(model, X, y, cv=5, method='predict_proba')
                meta_features[:, i] = cv_predictions[:, 1]  # Probability of class 1
            
            return meta_features
        
        # Create meta-learner
        meta_learner = LogisticRegression(random_state=random_state, max_iter=1000)
        
        # Store the stacking components
        self.ensemble_models['stacking'] = {
            'meta_learner': meta_learner,
            'create_meta_features': create_meta_features
        }
        
        return meta_learner
    
    def train_ensembles(self):
        """Train all ensemble models"""
        if self.base_predictor.X_train is None:
            print("No training data available.")
            return None
            
        results = {}
        
        # Create ensemble models
        print("Creating ensemble models...")
        self.create_voting_ensemble()
        self.create_bagging_ensemble()
        self.create_boosting_ensemble()
        self.create_stacking_ensemble()
        
        # Train each ensemble (except stacking which needs special handling)
        for name, model in self.ensemble_models.items():
            if name == 'stacking':
                continue  # Handle stacking separately
                
            print(f"\nTraining {name.upper()} ensemble...")
            
            # Train the model
            model.fit(self.base_predictor.X_train, self.base_predictor.y_train)
            
            # Make predictions
            y_pred = model.predict(self.base_predictor.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.base_predictor.y_test, y_pred)
            cv_scores = cross_val_score(model, self.base_predictor.X_train, self.base_predictor.y_train, cv=5)
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(self.base_predictor.y_test, y_pred),
                'confusion_matrix': confusion_matrix(self.base_predictor.y_test, y_pred)
            }
            
            print(f"{name.upper()} - Accuracy: {accuracy:.4f}")
            print(f"{name.upper()} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train stacking ensemble
        if 'stacking' in self.ensemble_models:
            print(f"\nTraining STACKING ensemble...")
            
            # Create meta-features for training
            meta_features_train = self.ensemble_models['stacking']['create_meta_features'](
                self.base_predictor.X_train, self.base_predictor.y_train
            )
            
            # Train meta-learner
            meta_learner = self.ensemble_models['stacking']['meta_learner']
            meta_learner.fit(meta_features_train, self.base_predictor.y_train)
            
            # Create meta-features for test set
            meta_features_test = np.zeros((self.base_predictor.X_test.shape[0], len(self.base_predictor.models)))
            for i, (name, model) in enumerate(self.base_predictor.models.items()):
                test_proba = model.predict_proba(self.base_predictor.X_test)
                meta_features_test[:, i] = test_proba[:, 1]
            
            # Make predictions
            y_pred = meta_learner.predict(meta_features_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.base_predictor.y_test, y_pred)
            
            # For CV score, we need to use the meta-features
            cv_scores = cross_val_score(meta_learner, meta_features_train, self.base_predictor.y_train, cv=5)
            
            results['stacking'] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(self.base_predictor.y_test, y_pred),
                'confusion_matrix': confusion_matrix(self.base_predictor.y_test, y_pred)
            }
            
            print(f"STACKING - Accuracy: {accuracy:.4f}")
            print(f"STACKING - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def predict_with_ensembles(self, input_data):
        """Make predictions using ensemble models"""
        # First get base model predictions
        base_predictions = self.base_predictor.predict_single(input_data)
        
        if not base_predictions:
            return None
            
        # Prepare input for ensemble models
        df_input = pd.DataFrame([input_data])
        
        # Process input same way as base models
        categorical_columns = [
            'gender', 'gpa', 'major_subjects', 'internship_grade', 'scholarship',
            'review_center', 'income_level', 'employment_status', 'employment_type',
            'parent_occup', 'parent_income'
        ]
        
        for col in categorical_columns:
            if col in df_input.columns and col in self.base_predictor.label_encoders:
                try:
                    df_input[col] = self.base_predictor.label_encoders[col].transform(df_input[col].astype(str))
                except ValueError:
                    df_input[col] = 0
        
        for feature in self.base_predictor.feature_names:
            if feature not in df_input.columns:
                df_input[feature] = 0
        
        df_input = df_input[self.base_predictor.feature_names]
        
        numerical_columns = ['age', 'mock_exam_score', 'test_anxiety', 'confidence', 
                           'study_hours', 'sleeping_hours']
        num_cols_present = [col for col in numerical_columns if col in df_input.columns]
        
        if num_cols_present:
            df_input[num_cols_present] = self.base_predictor.scaler.transform(df_input[num_cols_present])
        
        # Make ensemble predictions
        ensemble_predictions = {}
        
        for name, model in self.ensemble_models.items():
            try:
                if name == 'stacking':
                    # Special handling for stacking
                    meta_features = np.zeros((1, len(self.base_predictor.models)))
                    for i, (model_name, base_model) in enumerate(self.base_predictor.models.items()):
                        prob = base_model.predict_proba(df_input)
                        meta_features[0, i] = prob[0, 1]
                    
                    meta_learner = model['meta_learner']
                    pred = meta_learner.predict(meta_features)[0]
                    prob = meta_learner.predict_proba(meta_features)[0]
                else:
                    pred = model.predict(df_input)[0]
                    prob = model.predict_proba(df_input)[0] if hasattr(model, 'predict_proba') else None
                
                ensemble_predictions[name] = {
                    'prediction': int(pred),
                    'probability': prob.tolist() if prob is not None else None
                }
            except Exception as e:
                print(f"Error with {name} ensemble: {e}")
        
        # Combine base and ensemble predictions
        all_predictions = {
            'base_models': base_predictions,
            'ensemble_models': ensemble_predictions
        }
        
        return all_predictions
    
    def save_ensembles(self, models_dir='saved_ensemble_models'):
        """Save ensemble models"""
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        for name, model in self.ensemble_models.items():
            if name == 'stacking':
                # Save stacking components separately
                joblib.dump(model['meta_learner'], f'{models_dir}/stacking_meta_learner.joblib')
            else:
                joblib.dump(model, f'{models_dir}/{name}_ensemble.joblib')
        
        print(f"Ensemble models saved to {models_dir}/")
    
    def load_ensembles(self, models_dir='saved_ensemble_models'):
        """Load saved ensemble models"""
        try:
            ensemble_files = {
                'voting': 'voting_ensemble.joblib',
                'bagging': 'bagging_ensemble.joblib',
                'gradient_boosting': 'gradient_boosting_ensemble.joblib'
            }
            
            for name, file in ensemble_files.items():
                if os.path.exists(f'{models_dir}/{file}'):
                    self.ensemble_models[name] = joblib.load(f'{models_dir}/{file}')
            
            # Load stacking meta-learner
            stacking_file = f'{models_dir}/stacking_meta_learner.joblib'
            if os.path.exists(stacking_file):
                meta_learner = joblib.load(stacking_file)
                self.ensemble_models['stacking'] = {
                    'meta_learner': meta_learner,
                    'create_meta_features': self._create_meta_features_loaded
                }
            
            print(f"Ensemble models loaded from {models_dir}/")
            return True
            
        except Exception as e:
            print(f"Error loading ensemble models: {e}")
            return False
    
    def _create_meta_features_loaded(self, X, y=None):
        """Helper method for creating meta-features with loaded models"""
        meta_features = np.zeros((X.shape[0], len(self.base_predictor.models)))
        
        for i, (name, model) in enumerate(self.base_predictor.models.items()):
            proba = model.predict_proba(X)
            meta_features[:, i] = proba[:, 1]
        
        return meta_features

# Example usage
if __name__ == "__main__":
    # Load and train base models first
    base_predictor = SocialWorkPredictorModels()
    
    # Load sample data
    print("Loading sample data...")
    df = base_predictor.load_data_from_csv('social_work_sample_250.csv')
    
    if df is None:
        print("Creating sample data...")
        from base_models import create_sample_dataset
        df = create_sample_dataset(250, 'social_work_sample_250')
    
    X, y = base_predictor.preprocess_data(df)
    base_predictor.split_data(X, y)
    base_results = base_predictor.train_base_models()
    
    # Create and train ensemble models
    ensemble = EnsembleModels(base_predictor)
    ensemble_results = ensemble.train_ensembles()
    
    print("\n" + "="*70)
    print("COMPLETE MODEL COMPARISON (BASE + ENSEMBLE)")
    print("="*70)
    
    print("\nBASE MODELS:")
    for name, result in base_results.items():
        print(f"  {name.upper():<15}: {result['accuracy']:.4f}")
    
    print("\nENSEMBLE MODELS:")
    for name, result in ensemble_results.items():
        print(f"  {name.upper():<15}: {result['accuracy']:.4f}")
    
    # Find best model
    all_results = {**base_results, **ensemble_results}
    best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBEST MODEL: {best_model[0].upper()} with accuracy: {best_model[1]['accuracy']:.4f}")
    
    # Test combined predictions
    sample_input = {
        'age': 25, 'gender': 'female', 'gpa': 'VG', 'major_subjects': 'G',
        'internship_grade': 'G', 'scholarship': 'no', 'review_center': 'yes',
        'mock_exam_score': 4, 'test_anxiety': 3, 'confidence': 4,
        'study_hours': 8, 'sleeping_hours': 7, 'income_level': 'middle',
        'employment_status': 'unemployed', 'employment_type': 'regular',
        'parent_occup': 'skilled', 'parent_income': 'middle'
    }
    
    all_predictions = ensemble.predict_with_ensembles(sample_input)
    
    print("\n" + "="*70)
    print("SAMPLE PREDICTION - ALL MODELS")
    print("="*70)
    
    if all_predictions:
        print("\nBASE MODEL PREDICTIONS:")
        for model_name, result in all_predictions['base_models'].items():
            if result['probability']:
                pass_prob = result['probability'][1] * 100
                print(f"  {model_name.upper():<15}: {pass_prob:.1f}% chance of passing")
        
        print("\nENSEMBLE MODEL PREDICTIONS:")
        for model_name, result in all_predictions['ensemble_models'].items():
            if result['probability']:
                pass_prob = result['probability'][1] * 100
                print(f"  {model_name.upper():<15}: {pass_prob:.1f}% chance of passing")
    
    # Save all models
    base_predictor.save_models('saved_base_models')
    ensemble.save_ensembles('saved_ensemble_models')
    
    print(f"\nAll models saved successfully!")