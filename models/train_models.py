import os
import sys
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class PreprocessedModelTrainer:
    def __init__(self):
        self.models = {}
        self.ensemble_models = {}
        self.feature_names = []
        self.preprocessing_objects = None
        self.analysis_results = None
        
    def load_preprocessed_data(self, data_dir='processed_data', approach='onehot'):
        """Load preprocessed data from JSON files created by preprocessing.py"""
        try:
            # Check if processed_data directory exists
            if not os.path.exists(data_dir):
                print(f"❌ Error: {data_dir} directory not found!")
                print("   Please run preprocessing.py first to generate processed data.")
                return None
            
            # Load dataset JSON file
            dataset_file = f'{data_dir}/dataset_{approach}.json'
            if not os.path.exists(dataset_file):
                print(f"❌ Error: {dataset_file} not found!")
                print(f"   Available files: {os.listdir(data_dir)}")
                return None
                
            with open(dataset_file, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            X_train = np.array(data['X_train'])
            X_test = np.array(data['X_test'])
            y_train = np.array(data['y_train'])
            y_test = np.array(data['y_test'])
            feature_names = data['feature_names']
            
            # Load preprocessing objects
            preprocessing_file = f'{data_dir}/preprocessing_objects.pkl'
            if os.path.exists(preprocessing_file):
                self.preprocessing_objects = joblib.load(preprocessing_file)
            
            # Load analysis results
            analysis_file = f'{data_dir}/analysis_results.json'
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r') as f:
                    self.analysis_results = json.load(f)
            
            self.feature_names = feature_names
            
            print(f"✅ Loaded preprocessed data ({approach} approach)")
            print(f"   📊 Training samples: {X_train.shape[0]}")
            print(f"   📊 Test samples: {X_test.shape[0]}")
            print(f"   📊 Features: {X_train.shape[1]}")
            print(f"   📊 Feature names: {len(feature_names)}")
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': feature_names
            }
            
        except Exception as e:
            print(f"❌ Error loading preprocessed data: {e}")
            return None

    def display_data_info(self):
        """Display information about the loaded data"""
        if self.analysis_results:
            print("\n" + "="*60)
            print("📈 DATASET ANALYSIS SUMMARY")
            print("="*60)
            
            # Dataset info
            if 'data_stats' in self.analysis_results:
                stats = self.analysis_results['data_stats']
                print(f"\n📋 Dataset Statistics:")
                print(f"   Original Shape: {stats.get('shape', 'N/A')}")
                print(f"   Pass Rate: {stats.get('pass_rate', 0):.2%}")
                
                if 'target_distribution' in stats:
                    target_dist = stats['target_distribution']
                    print(f"   Target Distribution: {target_dist}")
            
            # Feature importance
            if 'feature_importance' in self.analysis_results:
                importance_data = self.analysis_results['feature_importance']
                if 'anova_f_test' in importance_data:
                    print(f"\n⭐ Top 10 Most Important Features (ANOVA F-test):")
                    f_scores = importance_data['anova_f_test']
                    sorted_features = sorted(f_scores.items(), key=lambda x: x[1], reverse=True)
                    for i, (feature, score) in enumerate(sorted_features[:10], 1):
                        print(f"   {i:2d}. {feature:<25}: {score:.4f}")

    def train_base_models(self, X_train, y_train, X_test, y_test):
        """Train base machine learning models"""
        print("\n" + "="*60)
        print("🤖 TRAINING BASE MODELS")
        print("="*60)
        
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
            print(f"\n🔧 Training {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
            
            # Store model and results
            self.models[name] = best_model
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'best_params': grid_search.best_params_,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"   ✅ {name:<20s}: Accuracy = {accuracy:.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"      Best params: {grid_search.best_params_}")
        
        return results

    def train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """Train ensemble models: Bagging, Boosting, Stacking"""
        print("\n" + "="*60)
        print("🎭 TRAINING ENSEMBLE MODELS")
        print("="*60)
        
        results = {}
        
        # 1. Bagging - Random Forest (inherently a bagging ensemble)
        print("\n🌳 Training Bagging Ensemble (Random Forest)...")
        bagging_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            bootstrap=True  # This makes it bagging
        )
        bagging_model.fit(X_train, y_train)
        bagging_pred = bagging_model.predict(X_test)
        bagging_acc = accuracy_score(y_test, bagging_pred)
        bagging_cv = cross_val_score(bagging_model, X_train, y_train, cv=5)
        
        self.ensemble_models['bagging_random_forest'] = bagging_model
        results['bagging_random_forest'] = {
            'accuracy': bagging_acc,
            'cv_mean': bagging_cv.mean(),
            'cv_std': bagging_cv.std(),
            'classification_report': classification_report(y_test, bagging_pred, output_dict=True)
        }
        print(f"   ✅ Bagging (Random Forest): {bagging_acc:.4f} (+/- {bagging_cv.std() * 2:.4f})")
        
        # 2. Boosting - Gradient Boosting
        print("\n🚀 Training Boosting Ensemble (Gradient Boosting)...")
        boosting_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        boosting_model.fit(X_train, y_train)
        boosting_pred = boosting_model.predict(X_test)
        boosting_acc = accuracy_score(y_test, boosting_pred)
        boosting_cv = cross_val_score(boosting_model, X_train, y_train, cv=5)
        
        self.ensemble_models['boosting_gradient_boost'] = boosting_model
        results['boosting_gradient_boost'] = {
            'accuracy': boosting_acc,
            'cv_mean': boosting_cv.mean(),
            'cv_std': boosting_cv.std(),
            'classification_report': classification_report(y_test, boosting_pred, output_dict=True)
        }
        print(f"   ✅ Boosting (Gradient Boost): {boosting_acc:.4f} (+/- {boosting_cv.std() * 2:.4f})")
        
        # 3. Stacking - with Logistic Regression as meta-learner
        print("\n🏗️  Training Stacking Ensemble (Logistic Regression meta-learner)...")
        
        # Use base models as level-0 estimators
        estimators = [(name, model) for name, model in self.models.items()]
        
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5,  # 5-fold cross-validation for training meta-learner
            random_state=42
        )
        stacking_model.fit(X_train, y_train)
        stacking_pred = stacking_model.predict(X_test)
        stacking_acc = accuracy_score(y_test, stacking_pred)
        stacking_cv = cross_val_score(stacking_model, X_train, y_train, cv=5)
        
        self.ensemble_models['stacking_logistic'] = stacking_model
        results['stacking_logistic'] = {
            'accuracy': stacking_acc,
            'cv_mean': stacking_cv.mean(),
            'cv_std': stacking_cv.std(),
            'classification_report': classification_report(y_test, stacking_pred, output_dict=True)
        }
        print(f"   ✅ Stacking (Logistic Reg): {stacking_acc:.4f} (+/- {stacking_cv.std() * 2:.4f})")
        
        return results

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

    def save_trained_models(self, directory='saved_models'):
        """Save all trained models"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save base models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(directory, f'{name}_model.pkl'))
        
        # Save ensemble models
        for name, model in self.ensemble_models.items():
            joblib.dump(model, os.path.join(directory, f'{name}_ensemble.pkl'))
        
        # Save feature names
        joblib.dump(self.feature_names, os.path.join(directory, 'feature_names.pkl'))
        
        # Save preprocessing objects if available
        if self.preprocessing_objects:
            joblib.dump(self.preprocessing_objects, os.path.join(directory, 'preprocessing_objects.pkl'))
        
        print(f"\n💾 All models saved to '{directory}' directory")
        print(f"   📄 Base models: {len(self.models)} files")
        print(f"   📄 Ensemble models: {len(self.ensemble_models)} files")
        print(f"   📄 Additional files: feature_names.pkl, preprocessing_objects.pkl")

    def predict_with_all_models(self, input_data_processed):
        """Make predictions with all trained models"""
        predictions = {}
        
        # Base model predictions
        for name, model in self.models.items():
            pred = model.predict(input_data_processed)[0]
            pred_proba = model.predict_proba(input_data_processed)[0] if hasattr(model, 'predict_proba') else None
            
            predictions[name] = {
                'prediction': int(pred),
                'probability': pred_proba.tolist() if pred_proba is not None else None
            }
        
        # Ensemble model predictions
        for name, model in self.ensemble_models.items():
            pred = model.predict(input_data_processed)[0]
            pred_proba = model.predict_proba(input_data_processed)[0] if hasattr(model, 'predict_proba') else None
            
            predictions[name] = {
                'prediction': int(pred),
                'probability': pred_proba.tolist() if pred_proba is not None else None
            }
        
        return predictions

def main():
    """Main training function using preprocessed data"""
    print("🚀 SOCIAL WORK EXAM MODEL TRAINING WITH PREPROCESSED DATA")
    print("=" * 70)
    
    # Initialize trainer
    trainer = PreprocessedModelTrainer()
    
    # Check if preprocessing has been done
    if not os.path.exists('processed_data'):
        print("❌ Preprocessed data not found!")
        print("   Please run the following command first:")
        print("   python preprocessing.py")
        print("\n   This will:")
        print("   📊 Analyze your CSV data")
        print("   🔧 Handle missing values and outliers")
        print("   📈 Perform feature importance analysis")
        print("   💾 Create clean training datasets (OneHot & Label encoded)")
        return
    
    # Test both preprocessing approaches
    approaches = ['onehot', 'label']
    results_comparison = {}
    
    for approach in approaches:
        print(f"\n{'='*20} {approach.upper()} APPROACH {'='*20}")
        
        # Load preprocessed data
        data = trainer.load_preprocessed_data(approach=approach)
        if data is None:
            print(f"❌ Failed to load {approach} data, skipping...")
            continue
        
        # Display data analysis info (only once)
        if approach == 'onehot':
            trainer.display_data_info()
        
        # Train base models
        print(f"\n🎯 Training base models with {approach} encoding...")
        base_results = trainer.train_base_models(
            data['X_train'], data['y_train'], data['X_test'], data['y_test']
        )
        
        # Train ensemble models (requires base models)
        print(f"\n🎭 Training ensemble models...")
        ensemble_results = trainer.train_ensemble_models(
            data['X_train'], data['y_train'], data['X_test'], data['y_test']
        )
        
        # Store results for comparison
        results_comparison[approach] = {
            'base_results': base_results,
            'ensemble_results': ensemble_results,
            'trainer': trainer
        }
        
        # Display results for this approach
        print(f"\n📊 {approach.upper()} APPROACH RESULTS:")
        print("-" * 50)
        
        # Combined results
        all_results = {**base_results, **ensemble_results}
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print(f"\n{'Rank':<4} {'Model':<25} {'Accuracy':<12} {'CV Score':<15}")
        print("-" * 60)
        for i, (model_name, metrics) in enumerate(sorted_results, 1):
            cv_info = f"{metrics['cv_mean']:.4f}±{metrics['cv_std']:.3f}"
            print(f"{i:<4} {model_name:<25} {metrics['accuracy']:<12.4f} {cv_info:<15}")
        
        # Show feature importance
        feature_importance = trainer.get_feature_importance('random_forest')
        if feature_importance:
            print(f"\n⭐ Top 10 Important Features (Random Forest):")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
                print(f"   {i:2d}. {feature:<25}: {importance:.4f}")
        
        # Save models for this approach
        trainer.save_trained_models(f'saved_models_{approach}')
    
    # Compare approaches
    if len(results_comparison) > 1:
        print("\n" + "="*70)
        print("🏆 FINAL COMPARISON: ONEHOT vs LABEL ENCODING")
        print("="*70)
        
        comparison_data = []
        for approach in approaches:
            if approach in results_comparison:
                # Get best models from each approach
                base_results = results_comparison[approach]['base_results']
                ensemble_results = results_comparison[approach]['ensemble_results']
                all_results = {**base_results, **ensemble_results}
                
                best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
                best_base = max(base_results.items(), key=lambda x: x[1]['accuracy'])
                best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['accuracy'])
                
                comparison_data.append({
                    'approach': approach.upper(),
                    'best_overall': best_model,
                    'best_base': best_base,
                    'best_ensemble': best_ensemble
                })
        
        print(f"\n📈 Best Models by Approach:")
        for data in comparison_data:
            print(f"\n{data['approach']} Encoding:")
            print(f"   🏆 Best Overall: {data['best_overall'][0]} ({data['best_overall'][1]['accuracy']:.4f})")
            print(f"   🤖 Best Base: {data['best_base'][0]} ({data['best_base'][1]['accuracy']:.4f})")
            print(f"   🎭 Best Ensemble: {data['best_ensemble'][0]} ({data['best_ensemble'][1]['accuracy']:.4f})")
        
        # Determine overall winner
        if len(comparison_data) == 2:
            if comparison_data[0]['best_overall'][1]['accuracy'] > comparison_data[1]['best_overall'][1]['accuracy']:
                winner = comparison_data[0]
                loser = comparison_data[1]
            else:
                winner = comparison_data[1]
                loser = comparison_data[0]
            
            print(f"\n🎉 WINNER: {winner['approach']} Encoding")
            print(f"   Best Model: {winner['best_overall'][0]}")
            print(f"   Accuracy: {winner['best_overall'][1]['accuracy']:.4f}")
            accuracy_diff = winner['best_overall'][1]['accuracy'] - loser['best_overall'][1]['accuracy']
            print(f"   Improvement: +{accuracy_diff:.4f} ({accuracy_diff*100:.2f}%)")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    total_models = 0
    for approach, data in results_comparison.items():
        base_count = len(data['base_results'])
        ensemble_count = len(data['ensemble_results'])
        total_models += base_count + ensemble_count
        print(f"📊 {approach.upper()} approach: {base_count} base + {ensemble_count} ensemble = {base_count + ensemble_count} models")
    
    print(f"🎯 Total models trained: {total_models}")
    print(f"💾 Models saved in: saved_models_onehot/ and saved_models_label/")
    print(f"📁 Preprocessed data available in: processed_data/")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Review model performance above")
    print(f"   2. Choose the best performing approach (OneHot or Label)")
    print(f"   3. Use saved models for prediction in your Django app")
    print(f"   4. Check saved_models_[approach]/ directory for model files")

if __name__ == "__main__":
    main()