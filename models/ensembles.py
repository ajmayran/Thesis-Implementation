import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class EnsembleModels:
    def __init__(self, base_models_dict):
        """Initialize with trained base models from base_models.py"""
        self.base_models = base_models_dict
        self.ensemble_models = {}
        
    def create_bagging_ensemble(self):
        """Create bagging ensemble using Random Forest"""
        rf_bagging = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            bootstrap=True
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
        """Create stacking ensemble with base models + Logistic Regression meta-learner"""
        # Use the 3 base models as level-0 estimators
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5,
            random_state=42
        )
        self.ensemble_models['stacking_logistic'] = stacking_clf
        return stacking_clf
    
    def train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """Train all 3 ensemble models"""
        print("\n" + "="*60)
        print("[ENSEMBLE] TRAINING 3 ENSEMBLE MODELS")
        print("="*60)
        
        results = {}
        
        # Create ensemble models
        print("\n[CREATE] Creating ensemble models...")
        self.create_bagging_ensemble()
        self.create_boosting_ensemble()
        self.create_stacking_ensemble()
        
        # Train and evaluate each ensemble
        for name, model in self.ensemble_models.items():
            print(f"\n   Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            print(f"      {name}: {accuracy:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
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
    
    def save_ensemble_models(self, directory='saved_ensemble_models'):
        """Save ensemble models"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for name, model in self.ensemble_models.items():
            joblib.dump(model, os.path.join(directory, f'{name}_ensemble.pkl'))
        
        print(f"\n[SAVE] Ensemble models saved to {directory}/")
    
    def load_ensemble_models(self, directory='saved_ensemble_models'):
        """Load ensemble models"""
        try:
            ensemble_files = [f for f in os.listdir(directory) if f.endswith('_ensemble.pkl')]
            
            for file in ensemble_files:
                model_name = file.replace('_ensemble.pkl', '')
                model_path = os.path.join(directory, file)
                self.ensemble_models[model_name] = joblib.load(model_path)
            
            print(f"[LOAD] Loaded {len(self.ensemble_models)} ensemble models from {directory}/")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading ensemble models: {e}")
            return False

def main():
    """Train ensemble models using saved base models"""
    from base_models import SocialWorkPredictorModels
    
    print("="*60)
    print("[START] TRAINING ENSEMBLE MODELS")
    print("="*60)
    
    # Load base models first
    predictor = SocialWorkPredictorModels()
    
    # Try to load saved base models
    if not predictor.load_models('saved_base_models'):
        print("\n[ERROR] No saved base models found!")
        print("[INFO] Please run base_models.py first to train base models")
        return
    
    # Load preprocessed data
    data = predictor.load_preprocessed_data(data_dir='processed_data', approach='label')
    
    if data is None:
        print("\n[ERROR] Could not load preprocessed data!")
        return
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Create and train ensembles
    ensemble = EnsembleModels(predictor.models)
    ensemble_results = ensemble.train_ensemble_models(X_train, y_train, X_test, y_test)
    
    # Display results
    print("\n" + "="*60)
    print("[RESULTS] ENSEMBLE MODEL PERFORMANCE")
    print("="*60)
    
    sorted_results = sorted(ensemble_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Model':<30} {'Accuracy':<12} {'CV Score':<15}")
    print("-" * 65)
    for i, (model_name, metrics) in enumerate(sorted_results, 1):
        cv_info = f"{metrics['cv_mean']:.4f}±{metrics['cv_std']:.3f}"
        print(f"{i:<5} {model_name.upper():<30} {metrics['accuracy']:<12.4f} {cv_info:<15}")
    
    # Save ensemble models
    ensemble.save_ensemble_models()
    
    print(f"\n[COMPLETE] Ensemble training completed!")
    print(f"[OUTPUT] 3 ensemble models saved to saved_ensemble_models/")

if __name__ == "__main__":
    main()