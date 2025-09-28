import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class EnsembleModels:
    def __init__(self, base_models_dict):
        self.base_models = base_models_dict
        self.ensemble_models = {}
        
    def create_bagging_ensemble(self):
        """Create bagging ensemble using Random Forest"""
        # Random Forest is already a bagging ensemble of decision trees
        rf_bagging = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            bootstrap=True  # This makes it bagging
        )
        self.ensemble_models['bagging_random_forest'] = rf_bagging
        return rf_bagging
    
    def create_boosting_ensemble(self):
        """Create boosting ensemble using Gradient Boosting"""
        gb_boosting = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.ensemble_models['boosting_gradient_boost'] = gb_boosting
        return gb_boosting
    
    def create_stacking_ensemble(self):
        """Create stacking ensemble with Logistic Regression as meta-learner"""
        from sklearn.ensemble import StackingClassifier
        
        # Use your base models as level-0 estimators
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        # Logistic Regression as meta-learner (level-1)
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5,  # 5-fold cross-validation for training meta-learner
            random_state=42
        )
        self.ensemble_models['stacking_logistic'] = stacking_clf
        return stacking_clf
    
    def train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """Train all ensemble models"""
        results = {}
        
        # Create ensemble models
        print("Creating ensemble models...")
        self.create_bagging_ensemble()
        self.create_boosting_ensemble()
        self.create_stacking_ensemble()
        
        # Train and evaluate each ensemble
        for name, model in self.ensemble_models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def predict_with_ensembles(self, X):
        """Make predictions with all ensemble models"""
        predictions = {}
        
        for name, model in self.ensemble_models.items():
            pred = model.predict(X)
            pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            
            predictions[name] = {
                'prediction': pred,
                'probability': pred_proba
            }
        
        return predictions
    
    def save_ensemble_models(self, directory='saved_models'):
        """Save ensemble models"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for name, model in self.ensemble_models.items():
            joblib.dump(model, os.path.join(directory, f'{name}_ensemble.pkl'))
        
        print(f"Ensemble models saved to {directory}")
    
    def load_ensemble_models(self, directory='saved_models'):
        """Load ensemble models"""
        try:
            ensemble_files = [f for f in os.listdir(directory) if f.endswith('_ensemble.pkl')]
            
            for file in ensemble_files:
                model_name = file.replace('_ensemble.pkl', '')
                model_path = os.path.join(directory, file)
                self.ensemble_models[model_name] = joblib.load(model_path)
            
            print(f"Ensemble models loaded from {directory}")
            return True
            
        except Exception as e:
            print(f"Error loading ensemble models: {e}")
            return False

def main():
    """Main function to test ensemble models"""
    from base_models import SocialWorkPredictorModels
    
    # Load and train base models first
    predictor = SocialWorkPredictorModels()
    
    print("Loading data...")
    df = predictor.load_data_from_csv('social_work_exam_dataset.csv')
    
    if df is None:
        print("Failed to load data.")
        return
    
    # Preprocess data
    print("Preprocessing data...")
    X, y = predictor.preprocess_data(df)
    
    if X is None or y is None:
        print("Failed to preprocess data.")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    predictor.X_train, predictor.X_test = X_train, X_test
    predictor.y_train, predictor.y_test = y_train, y_test
    
    # Train base models
    print("Training base models...")
    base_results = predictor.train_base_models()
    
    if not base_results:
        print("Failed to train base models.")
        return
    
    # Create and train ensemble models
    print("\n" + "="*50)
    print("TRAINING ENSEMBLE MODELS")
    print("="*50)
    
    ensemble = EnsembleModels(predictor.models)
    ensemble_results = ensemble.train_ensemble_models(X_train, y_train, X_test, y_test)
    
    # Display results
    print("\n" + "="*50)
    print("ENSEMBLE RESULTS SUMMARY")
    print("="*50)
    
    all_results = {**base_results, **ensemble_results}
    
    # Sort by accuracy
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\nModel Performance Ranking:")
    print("-" * 60)
    print(f"{'Rank':<4} {'Model':<25} {'Accuracy':<10} {'CV Score':<15}")
    print("-" * 60)
    for i, (model_name, metrics) in enumerate(sorted_results, 1):
        print(f"{i:<4} {model_name:<25} {metrics['accuracy']:<10.4f} {metrics['cv_mean']:.4f}±{metrics['cv_std']:.3f}")
    
    # Save ensemble models
    print("\nSaving ensemble models...")
    ensemble.save_ensemble_models()
    
    print("\nEnsemble training completed successfully!")

if __name__ == "__main__":
    main()