import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class EnsembleModels:
    def __init__(self, base_models_dict):
        """Initialize with trained base models from base_models.py"""
        self.base_models = base_models_dict
        self.ensemble_models = {}
        self.y_test = None
        
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
            passthrough=True,
            n_jobs=-1
        )
        self.ensemble_models['stacking_logistic'] = stacking_clf
        return stacking_clf
    
    def train_ensemble_models_with_10fold(self, X_train, y_train, X_test, y_test):
        """Train all 3 ensemble models with 10-fold cross-validation"""
        print("\n" + "="*60)
        print("[ENSEMBLE] TRAINING 3 ENSEMBLE MODELS WITH 10-FOLD CV")
        print("="*60)
        
        self.y_test = y_test
        results = {}
        
        # Combine train and test for 10-fold CV
        X_full = np.vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])
        
        # 10-fold stratified cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Create ensemble models
        print("\n[CREATE] Creating ensemble models...")
        self.create_bagging_ensemble()
        self.create_boosting_ensemble()
        self.create_stacking_ensemble()
        
        # Train and evaluate each ensemble with 10-fold CV
        for name, model in self.ensemble_models.items():
            print(f"\n[MODEL] Training {name.replace('_', ' ').upper()} with 10-fold CV...")
            
            # Perform 10-fold cross-validation on full dataset
            fold_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='accuracy')
            
            # Calculate detailed statistics
            cv_mean = np.mean(fold_scores)
            cv_std = np.std(fold_scores)
            cv_min = np.min(fold_scores)
            cv_max = np.max(fold_scores)
            
            # Train on training set
            model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate test metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'accuracy': test_accuracy,
                'cv_10fold_mean': cv_mean,
                'cv_10fold_std': cv_std,
                'cv_10fold_min': cv_min,
                'cv_10fold_max': cv_max,
                'cv_10fold_scores': fold_scores.tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            print(f"   [RESULTS] {name.replace('_', ' ').upper()}")
            print(f"      10-Fold CV Mean: {cv_mean:.4f}")
            print(f"      10-Fold CV Std:  {cv_std:.4f}")
            print(f"      10-Fold Range:   [{cv_min:.4f}, {cv_max:.4f}]")
            print(f"      Test Accuracy:   {test_accuracy:.4f}")
        
        return results
    
    def visualize_10fold_results(self, results, output_dir='ensemble_comparison'):
        """Create visualizations for 10-fold cross-validation results"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        model_names = list(results.keys())
        colors = ['#f59e0b', '#10b981', '#8b5cf6']  # Orange, Green, Purple
        
        # Plot 1: Box plot of 10-fold scores
        fold_data = []
        for name in model_names:
            for fold_idx, score in enumerate(results[name]['cv_10fold_scores'], 1):
                fold_data.append({
                    'Model': name.replace('_', ' ').upper(),
                    'Fold': fold_idx,
                    'Accuracy': score
                })
        
        fold_df = pd.DataFrame(fold_data)
        
        box_positions = []
        for idx, name in enumerate(model_names):
            model_data = fold_df[fold_df['Model'] == name.replace('_', ' ').upper()]['Accuracy']
            bp = axes[0, 0].boxplot([model_data], positions=[idx], widths=0.6,
                                    patch_artist=True, 
                                    boxprops=dict(facecolor=colors[idx], alpha=0.7),
                                    medianprops=dict(color='black', linewidth=2))
            box_positions.append(idx)
        
        axes[0, 0].set_xticks(box_positions)
        axes[0, 0].set_xticklabels([name.replace('_', ' ').upper() for name in model_names], rotation=15, ha='right')
        axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('10-Fold Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0.0, 1.0])
        
        # Plot 2: Line plot showing each fold's performance
        for idx, name in enumerate(model_names):
            fold_scores = results[name]['cv_10fold_scores']
            axes[0, 1].plot(range(1, 11), fold_scores, marker='o', linewidth=2, 
                          label=name.replace('_', ' ').upper(), color=colors[idx])
        
        axes[0, 1].set_xlabel('Fold Number', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Accuracy Across 10 Folds', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].set_xticks(range(1, 11))
        axes[0, 1].set_ylim([0.0, 1.0])
        
        # Plot 3: CV Mean vs Test Accuracy comparison
        x = np.arange(len(model_names))
        width = 0.35
        
        cv_means = [results[name]['cv_10fold_mean'] for name in model_names]
        test_accs = [results[name]['accuracy'] for name in model_names]
        
        bars1 = axes[1, 0].bar(x - width/2, cv_means, width, label='10-Fold CV Mean', 
                              color=colors, alpha=0.8)
        bars2 = axes[1, 0].bar(x + width/2, test_accs, width, label='Test Accuracy', 
                              color=colors, alpha=0.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        axes[1, 0].set_xlabel('Ensemble Models', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('10-Fold CV Mean vs Test Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([name.replace('_', ' ').upper() for name in model_names], rotation=15, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_ylim([0.0, 1.0])
        
        # Plot 4: Standard deviation comparison
        cv_stds = [results[name]['cv_10fold_std'] for name in model_names]
        
        bars = axes[1, 1].bar([name.replace('_', ' ').upper() for name in model_names], cv_stds, 
                             color=colors, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        axes[1, 1].set_xlabel('Ensemble Models', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('10-Fold CV Standard Deviation (Lower is Better)', 
                           fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ensemble_10fold_cv_analysis.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] Ensemble 10-fold CV analysis: {output_dir}/ensemble_10fold_cv_analysis.png")
        plt.close()
        
        # Save detailed 10-fold results to CSV
        detailed_fold_data = []
        for name in model_names:
            for fold_idx, score in enumerate(results[name]['cv_10fold_scores'], 1):
                detailed_fold_data.append({
                    'Model': name.replace('_', ' ').upper(),
                    'Fold': fold_idx,
                    'Accuracy': score
                })
        
        detailed_df = pd.DataFrame(detailed_fold_data)
        detailed_df.to_csv(f'{output_dir}/ensemble_10fold_detailed_scores.csv', index=False)
        print(f"[SAVED] Ensemble 10-fold detailed scores: {output_dir}/ensemble_10fold_detailed_scores.csv")
    
    def print_10fold_summary(self, results):
        """Print detailed 10-fold cross-validation summary for ensembles"""
        print("\n" + "="*90)
        print("ENSEMBLE 10-FOLD CROSS-VALIDATION SUMMARY")
        print("="*90)
        
        print(f"\n{'Model':<30} {'CV Mean':<12} {'CV Std':<12} {'CV Min':<12} {'CV Max':<12} {'Test Acc':<12}")
        print("-"*90)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_10fold_mean'], reverse=True)
        
        for name, result in sorted_results:
            print(f"{name.replace('_', ' ').upper():<30} "
                  f"{result['cv_10fold_mean']:<12.4f} "
                  f"{result['cv_10fold_std']:<12.4f} "
                  f"{result['cv_10fold_min']:<12.4f} "
                  f"{result['cv_10fold_max']:<12.4f} "
                  f"{result['accuracy']:<12.4f}")
        
        print("\n" + "="*90)
        print("INTERPRETATION")
        print("="*90)
        print("• CV Mean: Average accuracy across 10 folds (higher is better)")
        print("• CV Std:  Consistency across folds (lower is better)")
        print("• CV Min:  Worst-case performance")
        print("• CV Max:  Best-case performance")
        print("• Test Acc: Performance on held-out test set")
        
        # Find most consistent model
        most_consistent = min(results.items(), key=lambda x: x[1]['cv_10fold_std'])
        print(f"\n🎯 Most Consistent Ensemble: {most_consistent[0].replace('_', ' ').upper()} "
              f"(Std: {most_consistent[1]['cv_10fold_std']:.4f})")
        
        # Find best average performance
        best_avg = max(results.items(), key=lambda x: x[1]['cv_10fold_mean'])
        print(f"🏆 Best Average Performance: {best_avg[0].replace('_', ' ').upper()} "
              f"(CV Mean: {best_avg[1]['cv_10fold_mean']:.4f})")
    
    def compare_ensemble_visualization(self, results, output_dir='ensemble_comparison'):
        """Create comprehensive visualizations comparing the 3 ensemble models"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Main Dashboard - 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Accuracy Comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name].get('cv_10fold_mean', results[name].get('cv_mean', 0)) for name in model_names]
        
        x_pos = np.arange(len(model_names))
        
        axes[0, 0].bar(x_pos, accuracies, alpha=0.8, color='#f59e0b', label='Test Accuracy')
        axes[0, 0].bar(x_pos + 0.35, cv_means, alpha=0.8, color='#10b981', width=0.35, label='10-Fold CV Mean')
        axes[0, 0].set_xlabel('Ensemble Models', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Ensemble Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x_pos + 0.175)
        axes[0, 0].set_xticklabels([name.replace('_', ' ').upper() for name in model_names], rotation=15, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Precision, Recall, F1-Score
        metrics_data = []
        for name in model_names:
            report = results[name]['classification_report']
            metrics_data.append({
                'Model': name,
                'Precision': report['1']['precision'],
                'Recall': report['1']['recall'],
                'F1-Score': report['1']['f1-score']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        x = np.arange(len(model_names))
        width = 0.25
        
        axes[0, 1].bar(x - width, metrics_df['Precision'], width, label='Precision', color='#8b5cf6')
        axes[0, 1].bar(x, metrics_df['Recall'], width, label='Recall', color='#f59e0b')
        axes[0, 1].bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='#10b981')
        axes[0, 1].set_xlabel('Ensemble Models', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Score', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Precision, Recall, F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([name.replace('_', ' ').upper() for name in model_names], rotation=15, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: ROC Curves
        for name in model_names:
            if results[name]['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, results[name]['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                axes[1, 0].plot(fpr, tpr, label=f'{name.replace("_", " ").upper()} (AUC = {roc_auc:.3f})', linewidth=2)
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        axes[1, 0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].legend(loc='lower right', fontsize=8)
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Cross-Validation Score Distribution
        cv_data = []
        for name in model_names:
            cv_data.append({
                'Model': name.replace('_', ' ').upper(),
                'CV Mean': results[name].get('cv_10fold_mean', results[name].get('cv_mean', 0)),
                'CV Std': results[name].get('cv_10fold_std', results[name].get('cv_std', 0))
            })
        
        cv_df = pd.DataFrame(cv_data)
        axes[1, 1].bar(cv_df['Model'], cv_df['CV Mean'], yerr=cv_df['CV Std'], 
                       capsize=5, color='#f59e0b', alpha=0.7)
        axes[1, 1].set_xlabel('Ensemble Models', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Cross-Validation Score', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Cross-Validation Performance', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticklabels(cv_df['Model'], rotation=15, ha='right')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ensemble_comparison_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] Ensemble comparison dashboard: {output_dir}/ensemble_comparison_dashboard.png")
        plt.close()
        
        # 2. Confusion Matrices (1x3 layout)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, name in enumerate(model_names):
            cm = np.array(results[name]['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[idx],
                       xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
            axes[idx].set_title(f'{name.replace("_", " ").upper()}\nAccuracy: {results[name]["accuracy"]:.3f}', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ensemble_confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] Ensemble confusion matrices: {output_dir}/ensemble_confusion_matrices.png")
        plt.close()
        
        # 3. Performance Summary Table
        summary_data = []
        for name in model_names:
            summary_data.append({
                'Model': name.replace('_', ' ').upper(),
                'Accuracy': f"{results[name]['accuracy']:.4f}",
                'CV Mean': f"{results[name].get('cv_10fold_mean', results[name].get('cv_mean', 0)):.4f}",
                'CV Std': f"{results[name].get('cv_10fold_std', results[name].get('cv_std', 0)):.4f}",
                'Precision': f"{results[name]['classification_report']['1']['precision']:.4f}",
                'Recall': f"{results[name]['classification_report']['1']['recall']:.4f}",
                'F1-Score': f"{results[name]['classification_report']['1']['f1-score']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{output_dir}/ensemble_performance_summary.csv', index=False)
        print(f"[SAVED] Ensemble performance summary: {output_dir}/ensemble_performance_summary.csv")
        
        return summary_df
    
    def create_ensemble_accuracy_comparison(self, results, output_dir='ensemble_comparison'):
        """Create focused accuracy comparison for ensemble models"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model_names = list(results.keys())
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Accuracy Bar Chart
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name].get('cv_10fold_mean', results[name].get('cv_mean', 0)) for name in model_names]
        
        colors = ['#f59e0b', '#10b981', '#8b5cf6']  # Orange, Green, Purple
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, accuracies, width, label='Test Accuracy', color=colors, alpha=0.8)
        bars2 = axes[0].bar(x + width/2, cv_means, width, label='10-Fold CV Mean', color=colors, alpha=0.5)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[0].set_xlabel('Ensemble Models', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_title('Ensemble Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([name.replace('_', ' ').upper() for name in model_names], rotation=15, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1.0])
        
        # Plot 2: Accuracy with Error Bars
        cv_stds = [results[name].get('cv_10fold_std', results[name].get('cv_std', 0)) for name in model_names]
        
        axes[1].bar([name.replace('_', ' ').upper() for name in model_names], accuracies, yerr=cv_stds, 
                   capsize=10, color=colors, alpha=0.8, ecolor='black', linewidth=2)
        
        # Add value labels
        for i, (acc, std) in enumerate(zip(accuracies, cv_stds)):
            axes[1].text(i, acc + std + 0.02, f'{acc:.3f}\n±{std:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
        
        axes[1].set_xlabel('Ensemble Models', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title('Accuracy with Standard Deviation', fontsize=14, fontweight='bold')
        axes[1].set_xticklabels([name.replace('_', ' ').upper() for name in model_names], rotation=15, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim([0, 1.0])
        
        # Plot 3: Metrics Radar Chart
        from math import pi
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        ax = plt.subplot(133, projection='polar')
        
        for idx, (name, result) in enumerate(results.items()):
            report = result['classification_report']['1']
            values = [
                result['accuracy'],
                report['precision'],
                report['recall'],
                report['f1-score']
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=name.replace('_', ' ').upper(), color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Comparison', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ensemble_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] Ensemble accuracy comparison: {output_dir}/ensemble_accuracy_comparison.png")
        plt.close()
        
        # Detailed comparison table
        print(f"\n{'='*90}")
        print(f"3 ENSEMBLE MODELS - DETAILED ACCURACY COMPARISON (10-FOLD CV)")
        print(f"{'='*90}")
        print(f"\n{'Rank':<5} {'Model':<30} {'Test Acc':<12} {'CV Mean':<12} {'CV Std':<10} {'F1-Score':<10}")
        print(f"{'-'*90}")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            f1 = result['classification_report']['1']['f1-score']
            cv_mean = result.get('cv_10fold_mean', result.get('cv_mean', 0))
            cv_std = result.get('cv_10fold_std', result.get('cv_std', 0))
            print(f"{rank:<5} {name.replace('_', ' ').upper():<30} {result['accuracy']:<12.4f} "
                  f"{cv_mean:<12.4f} {cv_std:<10.4f} {f1:<10.4f}")
        
        # Statistical comparison
        print(f"\n{'='*90}")
        print(f"STATISTICAL COMPARISON")
        print(f"{'='*90}")
        
        winner = sorted_results[0]
        runner_up = sorted_results[1] if len(sorted_results) > 1 else None
        third = sorted_results[2] if len(sorted_results) > 2 else None
        
        print(f"\n🏆 1st Place: {winner[0].replace('_', ' ').upper()}")
        print(f"   Accuracy: {winner[1]['accuracy']:.4f}")
        
        if runner_up:
            acc_diff_1_2 = winner[1]['accuracy'] - runner_up[1]['accuracy']
            print(f"\n🥈 2nd Place: {runner_up[0].replace('_', ' ').upper()}")
            print(f"   Accuracy: {runner_up[1]['accuracy']:.4f}")
            print(f"   Difference from 1st: {acc_diff_1_2:.4f} ({acc_diff_1_2*100:.2f}%)")
        
        if third:
            acc_diff_1_3 = winner[1]['accuracy'] - third[1]['accuracy']
            print(f"\n🥉 3rd Place: {third[0].replace('_', ' ').upper()}")
            print(f"   Accuracy: {third[1]['accuracy']:.4f}")
            print(f"   Difference from 1st: {acc_diff_1_3:.4f} ({acc_diff_1_3*100:.2f}%)")
        
        return results
    
    def evaluate_ensemble_predictions(self, results, output_dir='ensemble_comparison'):
        """Detailed evaluation of ensemble predictions"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("\n" + "="*60)
        print("[EVALUATION] DETAILED ENSEMBLE PREDICTIONS ANALYSIS")
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
            
            print(f"\n[MODEL] {model_name.replace('_', ' ').upper()}")
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
    
    def create_ensemble_prediction_comparison(self, results, output_dir='ensemble_comparison'):
        """Visualize ensemble prediction accuracy"""
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
                            alpha=0.6, label='Predicted', color='orange', s=30, marker='x')
            
            # Highlight incorrect predictions
            incorrect_mask = self.y_test != y_pred
            if incorrect_mask.sum() > 0:
                axes[idx].scatter(comparison_df[incorrect_mask]['Index'], 
                                comparison_df[incorrect_mask]['Actual'],
                                color='red', s=100, marker='o', 
                                facecolors='none', linewidths=2, 
                                label='Incorrect')
            
            axes[idx].set_xlabel('Test Sample Index', fontsize=10)
            axes[idx].set_ylabel('Class (0=Fail, 1=Pass)', fontsize=10)
            axes[idx].set_title(f'{model_name.replace("_", " ").upper()}\nAccuracy: {result["accuracy"]:.3f}', 
                              fontsize=12, fontweight='bold')
            axes[idx].legend(loc='upper right')
            axes[idx].grid(alpha=0.3)
            axes[idx].set_ylim(-0.2, 1.2)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ensemble_prediction_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] Ensemble prediction comparison: {output_dir}/ensemble_prediction_comparison.png")
        plt.close()
    
    def generate_ensemble_report(self, results, output_dir='ensemble_comparison'):
        """Generate comprehensive ensemble test report"""
        report_path = f'{output_dir}/ensemble_evaluation_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Ensemble Models Testing and Evaluation Report (10-Fold Cross-Validation)\n\n")
            f.write(f"## Test Dataset Information\n\n")
            f.write(f"- **Total Test Samples:** {len(self.y_test)}\n")
            f.write(f"- **Actual Pass Count:** {(self.y_test == 1).sum()}\n")
            f.write(f"- **Actual Fail Count:** {(self.y_test == 0).sum()}\n")
            f.write(f"- **Pass Rate:** {(self.y_test == 1).mean():.2%}\n\n")
            
            f.write("## Ensemble Model Performance on Test Data\n\n")
            
            for model_name, result in sorted(results.items(), 
                                           key=lambda x: x[1]['accuracy'], reverse=True):
                y_pred = result['y_pred']
                cm = confusion_matrix(self.y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                f.write(f"### {model_name.replace('_', ' ').upper()}\n\n")
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
                f.write("---\n\n")
            
            f.write("## 10-Fold Cross-Validation Explained\n\n")
            f.write("10-fold cross-validation provides a robust estimate of model performance by:\n\n")
            f.write("1. Splitting the dataset into 10 equal parts (folds)\n")
            f.write("2. Training on 9 folds and testing on 1 fold\n")
            f.write("3. Repeating this process 10 times (each fold serves as test set once)\n")
            f.write("4. Averaging the results for final performance metrics\n\n")
            
            f.write("This approach reduces variance and provides more reliable performance estimates.\n\n")
            
            f.write("## Ensemble Techniques Used\n\n")
            f.write("### 1. Bagging (Random Forest)\n")
            f.write("- Multiple decision trees trained on bootstrap samples\n")
            f.write("- Reduces variance and prevents overfitting\n\n")
            
            f.write("### 2. Boosting (Gradient Boosting)\n")
            f.write("- Sequential training where each model corrects previous errors\n")
            f.write("- Reduces bias and improves accuracy\n\n")
            
            f.write("### 3. Stacking (with Logistic Regression)\n")
            f.write("- Combines predictions from 3 base models (KNN, Decision Tree, Random Forest)\n")
            f.write("- Meta-learner (Logistic Regression) learns optimal combination\n\n")
            
            f.write("## How to Interpret Results\n\n")
            f.write("- **True Positive:** Model correctly predicted student will PASS\n")
            f.write("- **True Negative:** Model correctly predicted student will FAIL\n")
            f.write("- **False Positive:** Model predicted PASS but student actually FAILED (Type I Error)\n")
            f.write("- **False Negative:** Model predicted FAIL but student actually PASSED (Type II Error)\n\n")
            
            f.write("## Testing Process\n\n")
            f.write("1. Ensemble models were validated using 10-fold cross-validation\n")
            f.write("2. Models were trained on 80% of data (training set)\n")
            f.write("3. Models make predictions on unseen 20% (test set)\n")
            f.write("4. Predictions are compared with actual results\n")
            f.write("5. Accuracy and other metrics are calculated\n")
        
        print(f"[SAVED] Ensemble evaluation report: {report_path}")
        return report_path
    
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
    """Train ensemble models using saved base models with 10-fold CV"""
    from base_models import SocialWorkPredictorModels
    
    print("="*60)
    print("[START] TRAINING ENSEMBLE MODELS WITH 10-FOLD CV")
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
    
    # Create and train ensembles with 10-fold CV
    ensemble = EnsembleModels(predictor.models)
    ensemble_results = ensemble.train_ensemble_models_with_10fold(X_train, y_train, X_test, y_test)
    
    # Print 10-fold summary
    ensemble.print_10fold_summary(ensemble_results)
    
    # Create 10-fold visualizations
    print(f"\n[VISUALIZATION] Creating 10-fold CV analysis...")
    ensemble.visualize_10fold_results(ensemble_results)
    
    # Display results
    print("\n" + "="*60)
    print("[RESULTS] ENSEMBLE MODEL PERFORMANCE")
    print("="*60)
    
    sorted_results = sorted(ensemble_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Model':<30} {'Accuracy':<12} {'CV Score':<15}")
    print("-" * 65)
    for i, (model_name, metrics) in enumerate(sorted_results, 1):
        cv_info = f"{metrics['cv_10fold_mean']:.4f}±{metrics['cv_10fold_std']:.3f}"
        print(f"{i:<5} {model_name.replace('_', ' ').upper():<30} {metrics['accuracy']:<12.4f} {cv_info:<15}")
    
    # Detailed evaluation
    ensemble.evaluate_ensemble_predictions(ensemble_results)
    
    # Create visualizations
    print(f"\n[VISUALIZATION] Creating ensemble comparison graphs...")
    summary_df = ensemble.compare_ensemble_visualization(ensemble_results)
    
    # Create accuracy comparison
    ensemble.create_ensemble_accuracy_comparison(ensemble_results)
    
    # Create prediction comparison plot
    ensemble.create_ensemble_prediction_comparison(ensemble_results)
    
    # Generate report
    ensemble.generate_ensemble_report(ensemble_results)
    
    print(f"\n[SUMMARY] Performance Summary Table:")
    print(summary_df.to_string(index=False))
    
    # Save ensemble models
    ensemble.save_ensemble_models()
    
    print(f"\n[COMPLETE] Ensemble training completed!")
    print(f"[OUTPUT] Files generated:")
    print(f"   - ensemble_comparison/ensemble_10fold_cv_analysis.png")
    print(f"   - ensemble_comparison/ensemble_10fold_detailed_scores.csv")
    print(f"   - ensemble_comparison/ensemble_comparison_dashboard.png")
    print(f"   - ensemble_comparison/ensemble_confusion_matrices.png")
    print(f"   - ensemble_comparison/ensemble_accuracy_comparison.png")
    print(f"   - ensemble_comparison/ensemble_prediction_comparison.png")
    print(f"   - ensemble_comparison/ensemble_evaluation_report.md")
    print(f"   - ensemble_comparison/*_test_predictions.csv (for each ensemble)")

if __name__ == "__main__":
    main()