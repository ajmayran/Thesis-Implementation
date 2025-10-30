import pandas as pd
import numpy as np
import json
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

class SocialWorkPredictorModels:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_names = []
        
        # Features available BEFORE taking the exam
        self.categorical_columns = ['Gender', 'IncomeLevel', 'EmploymentStatus']
        self.numerical_columns = ['Age', 'StudyHours', 'SleepHours', 'Confidence', 
                                'MockExamScore', 'GPA', 'Scholarship', 'InternshipGrade']
        self.binary_columns = ['ReviewCenter']
        
        self.target_column = 'Passed'
        
    def load_preprocessed_data(self, data_dir='processed_data', approach='label'):
        """Load preprocessed data from preprocessing.py output"""
        try:
            json_file = f'{data_dir}/dataset_{approach}.json'
            
            if not os.path.exists(json_file):
                print(f"[ERROR] Preprocessed data not found: {json_file}")
                print(f"[INFO] Please run preprocessing.py first")
                return None
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            X_train = np.array(data['X_train'])
            X_test = np.array(data['X_test'])
            y_train = np.array(data['y_train'])
            y_test = np.array(data['y_test'])
            feature_names = data['feature_names']
            
            self.feature_names = feature_names
            
            print(f"[SUCCESS] Loaded preprocessed data ({approach} approach)")
            print(f"   Training samples: {X_train.shape[0]}")
            print(f"   Test samples: {X_test.shape[0]}")
            print(f"   Features: {X_train.shape[1]}")
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': feature_names
            }
            
        except Exception as e:
            print(f"[ERROR] Error loading preprocessed data: {e}")
            return None
    
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
        """Train base models with 10-fold cross-validation"""
        if not hasattr(self, 'X_train') or self.X_train is None:
            print("[ERROR] Please load and preprocess data first")
            return None
        
        print(f"\n[TRAINING] Training base models with 10-FOLD CROSS-VALIDATION")
        print(f"[INFO] Using {self.X_train.shape[1]} features")
        print("="*70)
        
        results = {}
        
        # Combine train and test for 10-fold CV
        X_full = np.vstack([self.X_train, self.X_test])
        y_full = np.concatenate([self.y_train, self.y_test])
        
        # 10-fold stratified cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Model configurations (3 base models only)
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
                        'svm': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01]
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
            print(f"\n[MODEL] Training {name.upper()} with 10-fold CV...")
            
            # Grid search with 10-fold CV
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=cv,  # 10-fold stratified CV
                scoring='accuracy',
                n_jobs=-1,
                return_train_score=True
            )
            
            # Fit on full dataset
            grid_search.fit(X_full, y_full)
            best_model = grid_search.best_estimator_
            
            # Get 10-fold CV results
            cv_results = grid_search.cv_results_
            best_index = grid_search.best_index_
            
            # Store all 10 fold scores
            fold_scores = []
            for fold_idx in range(10):
                fold_key = f'split{fold_idx}_test_score'
                fold_scores.append(cv_results[fold_key][best_index])
            
            # Calculate detailed statistics
            cv_mean = np.mean(fold_scores)
            cv_std = np.std(fold_scores)
            cv_min = np.min(fold_scores)
            cv_max = np.max(fold_scores)
            
            # For final evaluation, use the original test set
            y_pred = best_model.predict(self.X_test)
            y_pred_proba = best_model.predict_proba(self.X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            test_accuracy = accuracy_score(self.y_test, y_pred)
            
            # Store results
            self.models[name] = best_model
            results[name] = {
                'accuracy': test_accuracy,
                'cv_10fold_mean': cv_mean,
                'cv_10fold_std': cv_std,
                'cv_10fold_min': cv_min,
                'cv_10fold_max': cv_max,
                'cv_10fold_scores': fold_scores,
                'best_params': grid_search.best_params_,
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
            }
            
            print(f"   [RESULTS] {name.upper()}")
            print(f"      10-Fold CV Mean: {cv_mean:.4f}")
            print(f"      10-Fold CV Std:  {cv_std:.4f}")
            print(f"      10-Fold Range:   [{cv_min:.4f}, {cv_max:.4f}]")
            print(f"      Test Accuracy:   {test_accuracy:.4f}")
            print(f"      Best Params:     {grid_search.best_params_}")
        
        return results
    
    def visualize_10fold_results(self, results, output_dir='model_comparison'):
        """Create visualizations for 10-fold cross-validation results"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        model_names = list(results.keys())
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
        
        fold_data = []
        for name in model_names:
            for fold_idx, score in enumerate(results[name]['cv_10fold_scores'], 1):
                fold_data.append({
                    'Model': name.upper(),
                    'Fold': fold_idx,
                    'Accuracy': score
                })
        
        fold_df = pd.DataFrame(fold_data)
        
        box_positions = []
        for idx, name in enumerate(model_names):
            model_data = fold_df[fold_df['Model'] == name.upper()]['Accuracy']
            bp = axes[0, 0].boxplot([model_data], positions=[idx], widths=0.6,
                                    patch_artist=True, 
                                    boxprops=dict(facecolor=colors[idx], alpha=0.7),
                                    medianprops=dict(color='black', linewidth=2))
            box_positions.append(idx)
        
        axes[0, 0].set_xticks(box_positions)
        axes[0, 0].set_xticklabels([name.upper() for name in model_names], rotation=45, ha='right')
        axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('10-Fold Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0.0, 1.0])
        
        for idx, name in enumerate(model_names):
            fold_scores = results[name]['cv_10fold_scores']
            axes[0, 1].plot(range(1, 11), fold_scores, marker='o', linewidth=2, 
                          label=name.upper(), color=colors[idx])
        
        axes[0, 1].set_xlabel('Fold Number', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Accuracy Across 10 Folds', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].set_xticks(range(1, 11))
        axes[0, 1].set_ylim([0.0, 1.0])
        
        x = np.arange(len(model_names))
        width = 0.35
        
        cv_means = [results[name]['cv_10fold_mean'] for name in model_names]
        test_accs = [results[name]['accuracy'] for name in model_names]
        
        bars1 = axes[1, 0].bar(x - width/2, cv_means, width, label='10-Fold CV Mean', 
                              color=colors[:len(model_names)], alpha=0.8)
        bars2 = axes[1, 0].bar(x + width/2, test_accs, width, label='Test Accuracy', 
                              color=colors[:len(model_names)], alpha=0.5)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        axes[1, 0].set_xlabel('Models', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('10-Fold CV Mean vs Test Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([name.upper() for name in model_names], rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_ylim([0.0, 1.0])
        
        cv_stds = [results[name]['cv_10fold_std'] for name in model_names]
        
        bars = axes[1, 1].bar([name.upper() for name in model_names], cv_stds, 
                             color=colors[:len(model_names)], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        axes[1, 1].set_xlabel('Models', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('10-Fold CV Standard Deviation (Lower is Better)', 
                           fontsize=14, fontweight='bold')
        axes[1, 1].set_xticklabels([name.upper() for name in model_names], rotation=45, ha='right')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/10fold_cv_analysis.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] 10-fold CV analysis: {output_dir}/10fold_cv_analysis.png")
        plt.close()
        
        detailed_fold_data = []
        for name in model_names:
            for fold_idx, score in enumerate(results[name]['cv_10fold_scores'], 1):
                detailed_fold_data.append({
                    'Model': name.upper(),
                    'Fold': fold_idx,
                    'Accuracy': score
                })
        
        detailed_df = pd.DataFrame(detailed_fold_data)
        detailed_df.to_csv(f'{output_dir}/10fold_detailed_scores.csv', index=False)
        print(f"[SAVED] 10-fold detailed scores: {output_dir}/10fold_detailed_scores.csv")
    
    def print_10fold_summary(self, results):
        """Print detailed 10-fold cross-validation summary"""
        print("\n" + "="*80)
        print("10-FOLD CROSS-VALIDATION SUMMARY")
        print("="*80)
        
        print(f"\n{'Model':<20} {'CV Mean':<12} {'CV Std':<12} {'CV Min':<12} {'CV Max':<12} {'Test Acc':<12}")
        print("-"*80)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_10fold_mean'], reverse=True)
        
        for name, result in sorted_results:
            print(f"{name.upper():<20} "
                  f"{result['cv_10fold_mean']:<12.4f} "
                  f"{result['cv_10fold_std']:<12.4f} "
                  f"{result['cv_10fold_min']:<12.4f} "
                  f"{result['cv_10fold_max']:<12.4f} "
                  f"{result['accuracy']:<12.4f}")
        
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        print("• CV Mean: Average accuracy across 10 folds (higher is better)")
        print("• CV Std:  Consistency across folds (lower is better)")
        print("• CV Min:  Worst-case performance")
        print("• CV Max:  Best-case performance")
        print("• Test Acc: Performance on held-out test set")
        
        # Find most consistent model
        most_consistent = min(results.items(), key=lambda x: x[1]['cv_10fold_std'])
        print(f"\n🎯 Most Consistent Model: {most_consistent[0].upper()} (Std: {most_consistent[1]['cv_10fold_std']:.4f})")
        
        # Find best average performance
        best_avg = max(results.items(), key=lambda x: x[1]['cv_10fold_mean'])
        print(f"🏆 Best Average Performance: {best_avg[0].upper()} (CV Mean: {best_avg[1]['cv_10fold_mean']:.4f})")
    
    def compare_models_visualization(self, results, output_dir='model_comparison'):
        """Create comprehensive visualizations comparing the 6 models"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_names = list(results.keys())
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name]['cv_10fold_mean'] for name in model_names]
        
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, accuracies, width, label='Test Accuracy', color=colors, alpha=0.8)
        axes[0, 0].bar(x_pos + width/2, cv_means, width, label='CV Mean', color=colors, alpha=0.5)
        axes[0, 0].set_xlabel('Models', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([name.upper() for name in model_names], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        for i, (acc, cv) in enumerate(zip(accuracies, cv_means)):
            axes[0, 0].text(i - width/2, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
            axes[0, 0].text(i + width/2, cv + 0.02, f'{cv:.3f}', ha='center', va='bottom', fontsize=9)
        
        for idx, name in enumerate(model_names):
            cm = np.array(results[name]['confusion_matrix'])
            axes[0, 1].text(0.1, 0.9 - idx*0.15, f"{name.upper()}: {results[name]['accuracy']:.3f}", 
                        transform=axes[0, 1].transAxes, fontsize=10, fontweight='bold')
        axes[0, 1].axis('off')
        axes[0, 1].set_title('Model Accuracy Summary', fontsize=14, fontweight='bold')
        
        for name in model_names:
            if results[name]['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, results[name]['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                axes[1, 0].plot(fpr, tpr, label=f'{name.upper()} (AUC={roc_auc:.3f})', linewidth=2)
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        axes[1, 0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('ROC Curves', fontsize=14, fontweight='bold')
        axes[1, 0].legend(loc='lower right')
        axes[1, 0].grid(alpha=0.3)
        
        cv_stds = [results[name]['cv_10fold_std'] for name in model_names]
        axes[1, 1].bar(range(len(model_names)), cv_stds, color=colors, alpha=0.8)
        axes[1, 1].set_xlabel('Models', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('CV Standard Deviation', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Model Stability (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(range(len(model_names)))
        axes[1, 1].set_xticklabels([name.upper() for name in model_names], rotation=45, ha='right')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] Model comparison dashboard: {output_dir}/model_comparison_dashboard.png")
        plt.close()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, name in enumerate(model_names):
            cm = np.array(results[name]['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
            axes[idx].set_title(f'{name.upper()}\nAccuracy: {results[name]["accuracy"]:.3f}', 
                            fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] Confusion matrices: {output_dir}/confusion_matrices.png")
        plt.close()
        
        summary_data = []
        for name in model_names:
            summary_data.append({
                'Model': name.upper(),
                'Accuracy': f"{results[name]['accuracy']:.4f}",
                'CV Mean': f"{results[name]['cv_10fold_mean']:.4f}",
                'CV Std': f"{results[name]['cv_10fold_std']:.4f}",
                'Precision': f"{results[name]['classification_report']['1']['precision']:.4f}",
                'Recall': f"{results[name]['classification_report']['1']['recall']:.4f}",
                'F1-Score': f"{results[name]['classification_report']['1']['f1-score']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{output_dir}/model_performance_summary.csv', index=False)
        print(f"[SAVED] Performance summary: {output_dir}/model_performance_summary.csv")
        
        return summary_df

    def create_top_accuracy_comparison(self, results, output_dir='model_comparison'):
        """Create focused accuracy comparison for all models"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_names = list(results.keys())
        
        plt.figure(figsize=(16, 12))
        
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name].get('cv_10fold_mean', results[name].get('cv_mean', 0)) for name in model_names]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

        x = np.arange(len(model_names))
        width = 0.35
        
        ax1 = plt.subplot(2, 2, 1)
        ax1.bar(x - width/2, accuracies, width, label='Test Accuracy', color=colors, alpha=0.8)
        ax1.bar(x + width/2, cv_means, width, label='10-Fold CV Mean', color=colors, alpha=0.5)
        
        for i, acc in enumerate(accuracies):
            ax1.text(i - width/2, acc, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([name.upper() for name in model_names])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.0])
        
        cv_stds = [results[name].get('cv_10fold_std', results[name].get('cv_std', 0)) for name in model_names]
        
        ax2 = plt.subplot(2, 2, 2)
        ax2.bar(model_names, accuracies, yerr=cv_stds, capsize=10, 
                   color=colors, alpha=0.8, ecolor='black', linewidth=2)
        
        for i, (acc, std) in enumerate(zip(accuracies, cv_stds)):
            ax2.text(i, acc + std + 0.02, f'{acc:.3f}\n±{std:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Accuracy with Standard Deviation', fontsize=14, fontweight='bold')
        ax2.set_xticklabels([name.upper() for name in model_names])
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 1.0])
        
        from math import pi
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        ax = plt.subplot(2, 2, 3, projection='polar')
        
        for idx, (name, result) in enumerate(results.items()):
            report = result['classification_report']['1']
            values = [
                result['accuracy'],
                report['precision'],
                report['recall'],
                report['f1-score']
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=name.upper(), color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Comparison', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top3_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] 3-Model accuracy comparison: {output_dir}/top3_accuracy_comparison.png")
        plt.close()
        
        # Create a detailed comparison table
        print(f"\n{'='*70}")
        print(f"3 BASE MODELS - DETAILED ACCURACY COMPARISON (10-FOLD CV)")
        print(f"{'='*70}")
        print(f"\n{'Rank':<5} {'Model':<20} {'Test Acc':<12} {'CV Mean':<12} {'CV Std':<10} {'F1-Score':<10}")
        print(f"{'-'*70}")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            f1 = result['classification_report']['1']['f1-score']
            cv_mean = result.get('cv_10fold_mean', result.get('cv_mean', 0))
            cv_std = result.get('cv_10fold_std', result.get('cv_std', 0))
            print(f"{rank:<5} {name.upper():<20} {result['accuracy']:<12.4f} "
                  f"{cv_mean:<12.4f} {cv_std:<10.4f} {f1:<10.4f}")
        
        return results
    
    def predict_single(self, input_data):
        """Make prediction for a single candidate BEFORE they take the exam"""
        try:
            # Map input data to correct feature names
            if isinstance(input_data, dict):
                mapped_data = {}
                
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
        
    def evaluate_test_predictions(self, results, output_dir='model_comparison'):
        """Detailed evaluation of test predictions showing correct/incorrect predictions"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("\n" + "="*60)
        print("[EVALUATION] DETAILED TEST PREDICTIONS ANALYSIS")
        print("="*60)
        
        for model_name, result in results.items():
            y_pred = result['y_pred']
            
            # Create detailed prediction DataFrame
            prediction_details = pd.DataFrame({
                'Actual': self.y_test,
                'Predicted': y_pred,
                'Correct': self.y_test == y_pred,
                'Actual_Label': ['Pass' if y == 1 else 'Fail' for y in self.y_test],
                'Predicted_Label': ['Pass' if y == 1 else 'Fail' for y in y_pred]
            })
            
            # Add probability if available
            if result['y_pred_proba'] is not None:
                prediction_details['Pass_Probability'] = result['y_pred_proba']
            
            # Summary statistics
            total_samples = len(self.y_test)
            correct_predictions = (self.y_test == y_pred).sum()
            incorrect_predictions = total_samples - correct_predictions
            
            print(f"\n[MODEL] {model_name.upper()}")
            print(f"   Total Test Samples: {total_samples}")
            print(f"   Correct Predictions: {correct_predictions} ({correct_predictions/total_samples*100:.2f}%)")
            print(f"   Incorrect Predictions: {incorrect_predictions} ({incorrect_predictions/total_samples*100:.2f}%)")
            
            # Breakdown by class
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            print(f"\n   [BREAKDOWN]")
            print(f"   True Positives (Predicted Pass, Actually Pass): {tp}")
            print(f"   True Negatives (Predicted Fail, Actually Fail): {tn}")
            print(f"   False Positives (Predicted Pass, Actually Fail): {fp}")
            print(f"   False Negatives (Predicted Fail, Actually Pass): {fn}")
            
            # Save detailed predictions to CSV
            csv_file = f'{output_dir}/{model_name}_test_predictions.csv'
            prediction_details.to_csv(csv_file, index=True)
            print(f"\n   [SAVED] Detailed predictions: {csv_file}")
            
            # Show some examples
            print(f"\n   [EXAMPLES] First 10 predictions:")
            print(prediction_details.head(10).to_string(index=True))
            
            # Show incorrect predictions
            incorrect = prediction_details[~prediction_details['Correct']]
            if len(incorrect) > 0:
                print(f"\n   [ERRORS] First 5 incorrect predictions:")
                print(incorrect.head(5).to_string(index=True))
    
    def create_prediction_comparison_plot(self, results, output_dir='model_comparison'):
        """Visualize prediction accuracy for each model"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, (model_name, result) in enumerate(results.items()):
            y_pred = result['y_pred']
            
            # Create comparison plot
            comparison_df = pd.DataFrame({
                'Index': range(len(self.y_test)),
                'Actual': self.y_test,
                'Predicted': y_pred
            })
            
            # Plot actual vs predicted
            axes[idx].scatter(comparison_df['Index'], comparison_df['Actual'], 
                            alpha=0.6, label='Actual', color='blue', s=50)
            axes[idx].scatter(comparison_df['Index'], comparison_df['Predicted'], 
                            alpha=0.6, label='Predicted', color='red', s=30, marker='x')
            
            # Highlight incorrect predictions
            incorrect_mask = self.y_test != y_pred
            if incorrect_mask.sum() > 0:
                axes[idx].scatter(comparison_df[incorrect_mask]['Index'], 
                                comparison_df[incorrect_mask]['Actual'],
                                color='orange', s=100, marker='o', 
                                facecolors='none', linewidths=2, 
                                label='Incorrect')
            
            axes[idx].set_xlabel('Test Sample Index', fontsize=10)
            axes[idx].set_ylabel('Class (0=Fail, 1=Pass)', fontsize=10)
            axes[idx].set_title(f'{model_name.upper()}\nAccuracy: {result["accuracy"]:.3f}', 
                              fontsize=12, fontweight='bold')
            axes[idx].legend(loc='upper right')
            axes[idx].grid(alpha=0.3)
            axes[idx].set_ylim(-0.2, 1.2)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/prediction_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] Prediction comparison plot: {output_dir}/prediction_comparison.png")
        plt.close()
    
    def generate_test_report(self, results, output_dir='model_comparison'):
        """Generate comprehensive test report in markdown"""
        report_path = f'{output_dir}/test_evaluation_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Model Testing and Evaluation Report (10-Fold Cross-Validation)\n\n")
            f.write(f"## Test Dataset Information\n\n")
            f.write(f"- **Total Test Samples:** {len(self.y_test)}\n")
            f.write(f"- **Actual Pass Count:** {(self.y_test == 1).sum()}\n")
            f.write(f"- **Actual Fail Count:** {(self.y_test == 0).sum()}\n")
            f.write(f"- **Pass Rate:** {(self.y_test == 1).mean():.2%}\n\n")
            
            f.write("## Model Performance on Test Data\n\n")
            
            for model_name, result in sorted(results.items(), 
                                           key=lambda x: x[1]['accuracy'], reverse=True):
                y_pred = result['y_pred']
                cm = confusion_matrix(self.y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                f.write(f"### {model_name.upper()}\n\n")
                f.write(f"**Overall Accuracy:** {result['accuracy']:.4f}\n\n")
                
                if 'cv_10fold_mean' in result:
                    f.write(f"**10-Fold Cross-Validation:**\n\n")
                    f.write(f"- **CV Mean:** {result['cv_10fold_mean']:.4f}\n")
                    f.write(f"- **CV Std:** {result['cv_10fold_std']:.4f}\n")
                    f.write(f"- **CV Min:** {result['cv_10fold_min']:.4f}\n")
                    f.write(f"- **CV Max:** {result['cv_10fold_max']:.4f}\n\n")
                
                f.write("**Confusion Matrix:**\n\n")
                f.write("```\n")
                f.write(f"                Predicted Fail    Predicted Pass\n")
                f.write(f"Actual Fail          {tn:4d}              {fp:4d}\n")
                f.write(f"Actual Pass          {fn:4d}              {tp:4d}\n")
                f.write("```\n\n")
                
                f.write("**Detailed Metrics:**\n\n")
                f.write(f"- **True Positives (Correctly predicted Pass):** {tp} ({tp/len(self.y_test)*100:.1f}%)\n")
                f.write(f"- **True Negatives (Correctly predicted Fail):** {tn} ({tn/len(self.y_test)*100:.1f}%)\n")
                f.write(f"- **False Positives (Wrongly predicted Pass):** {fp} ({fp/len(self.y_test)*100:.1f}%)\n")
                f.write(f"- **False Negatives (Wrongly predicted Fail):** {fn} ({fn/len(self.y_test)*100:.1f}%)\n\n")
                
                report = result['classification_report']
                f.write("**Classification Report:**\n\n")
                f.write(f"- **Precision (Pass class):** {report['1']['precision']:.4f}\n")
                f.write(f"- **Recall (Pass class):** {report['1']['recall']:.4f}\n")
                f.write(f"- **F1-Score (Pass class):** {report['1']['f1-score']:.4f}\n\n")
                
                f.write(f"**Best Hyperparameters:** {result['best_params']}\n\n")
                f.write("---\n\n")
            
            f.write("## 10-Fold Cross-Validation Explained\n\n")
            f.write("10-fold cross-validation provides a robust estimate of model performance by:\n\n")
            f.write("1. Splitting the dataset into 10 equal parts (folds)\n")
            f.write("2. Training on 9 folds and testing on 1 fold\n")
            f.write("3. Repeating this process 10 times (each fold serves as test set once)\n")
            f.write("4. Averaging the results for final performance metrics\n\n")
            
            f.write("This approach reduces variance and provides more reliable performance estimates.\n\n")
            
            f.write("## How to Interpret Results\n\n")
            f.write("- **True Positive:** Model correctly predicted student will PASS\n")
            f.write("- **True Negative:** Model correctly predicted student will FAIL\n")
            f.write("- **False Positive:** Model predicted PASS but student actually FAILED (Type I Error)\n")
            f.write("- **False Negative:** Model predicted FAIL but student actually PASSED (Type II Error)\n\n")
            
            f.write("## Testing Process\n\n")
            f.write("1. Models were trained using 10-fold cross-validation\n")
            f.write("2. Best hyperparameters were selected based on CV performance\n")
            f.write("3. Final models make predictions on held-out test set\n")
            f.write("4. Performance metrics are calculated and compared\n")
        
        print(f"[SAVED] Test evaluation report: {report_path}")
        return report_path
    
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
    """Main training function using preprocessed data with 10-fold CV"""
    predictor = SocialWorkPredictorModels()
    
    print("="*60)
    print("[START] TRAINING BASE MODELS WITH 10-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # Load preprocessed data
    data = predictor.load_preprocessed_data(data_dir='processed_data', approach='label')
    
    if data is None:
        print("\n[ERROR] Could not load preprocessed data!")
        print("[INFO] Please run: python preprocessing.py")
        return
    
    # Set data for training
    predictor.X_train = data['X_train']
    predictor.X_test = data['X_test']
    predictor.y_train = data['y_train']
    predictor.y_test = data['y_test']
    
    # Train 3 base models with 10-fold CV
    results = predictor.train_base_models()
    
    if results:
        # Print 10-fold summary
        predictor.print_10fold_summary(results)
        
        # Create 10-fold visualizations
        print(f"\n[VISUALIZATION] Creating 10-fold CV analysis...")
        predictor.visualize_10fold_results(results)
        
        # Original visualizations
        print(f"\n[VISUALIZATION] Creating model comparison graphs...")
        summary_df = predictor.compare_models_visualization(results)
        predictor.create_top_accuracy_comparison(results)
        
        # Detailed evaluation
        predictor.evaluate_test_predictions(results)
        predictor.create_prediction_comparison_plot(results)
        predictor.generate_test_report(results)
        
        print(f"\n[SUMMARY] Performance Summary Table:")
        print(summary_df.to_string(index=False))
        
        # Feature importance
        feature_importance = predictor.get_feature_importance('random_forest')
        if feature_importance:
            print(f"\n[FEATURE IMPORTANCE] Top 10 Predictive Factors:")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
                print(f"{i:2d}. {feature:<25s}: {importance:.4f}")
        
        # Save models
        predictor.save_models('saved_base_models')
        
        print(f"\n[COMPLETE] Training completed successfully!")
        print(f"[OUTPUT] Files generated:")
        print(f"   - model_comparison/10fold_cv_analysis.png")
        print(f"   - model_comparison/10fold_detailed_scores.csv")
        print(f"   - model_comparison/model_comparison_dashboard.png")
        print(f"   - model_comparison/confusion_matrices.png")
        print(f"   - model_comparison/top3_accuracy_comparison.png")
        print(f"   - model_comparison/prediction_comparison.png")
        print(f"   - model_comparison/test_evaluation_report.md")

if __name__ == "__main__":
    main()