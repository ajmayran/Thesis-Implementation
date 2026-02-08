import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import os
import warnings
import json
warnings.filterwarnings('ignore')

class EnsembleModels:
    def __init__(self, best_params_dict=None):
        self.best_params = best_params_dict or {}
        self.ensemble_models = {}
        self.y_test = None
        
    def create_bagging_ensemble(self):
        bagging = BaggingClassifier(
            estimator=RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            ),
            n_estimators=15,
            max_samples=0.7,
            max_features=0.9,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        self.ensemble_models['bagging'] = bagging
        print("[INFO] Bagging: 15 Random Forests with 70% samples")
        return bagging
    
    def create_boosting_ensemble(self):
        boosting = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        )
        self.ensemble_models['boosting'] = boosting
        print("[INFO] Boosting: 150 estimators, LR=0.05, depth=4")
        return boosting
    
    def create_stacking_ensemble(self):
        rf_params = self.best_params.get('random_forest', {'n_estimators': 100, 'max_depth': 15})
        svm_params = self.best_params.get('svm', {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'})
        dt_params = self.best_params.get('decision_tree', {'max_depth': 10, 'min_samples_split': 5})
        
        estimators = [
            ('random_forest', RandomForestClassifier(random_state=42, **rf_params)),
            ('svm', SVC(probability=True, random_state=42, **svm_params)),
            ('decision_tree', DecisionTreeClassifier(random_state=42, **dt_params))
        ]
        
        print(f"[INFO] Selective Stacking: Using only top 3 performing base models:")
        for name, _ in estimators:
            print(f"   - {name.replace('_', ' ').upper()}")
        
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5,
            n_jobs=-1
        )
        self.ensemble_models['stacking'] = stacking
        return stacking

    def create_weighted_voting_ensemble(self):
        rf_params = self.best_params.get('random_forest', {'n_estimators': 100, 'max_depth': 15})
        svm_params = self.best_params.get('svm', {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'})
        dt_params = self.best_params.get('decision_tree', {'max_depth': 10, 'min_samples_split': 5})
        
        from sklearn.ensemble import VotingClassifier
        
        estimators = [
            ('random_forest', RandomForestClassifier(random_state=42, **rf_params)),
            ('svm', SVC(probability=True, random_state=42, **svm_params)),
            ('decision_tree', DecisionTreeClassifier(random_state=42, **dt_params))
        ]
        
        voting = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[2, 1.5, 1],
            n_jobs=-1
        )
        
        self.ensemble_models['weighted_voting'] = voting
        print("[INFO] Weighted Voting: RF(2.0), SVM(1.5), DT(1.0)")
        return voting
    
    def train_ensemble_models_with_10fold(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*60)
        print("[ENSEMBLE] TRAINING ENSEMBLE MODELS WITH 10-FOLD CV")
        print("="*60)
        
        self.y_test = y_test
        results = {}
        
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        print("\n[CREATE] Creating ensemble models...")
        self.create_bagging_ensemble()
        self.create_boosting_ensemble()
        self.create_stacking_ensemble()
        self.create_weighted_voting_ensemble()
        
        for name, model in self.ensemble_models.items():
            print(f"\n[MODEL] Training {name.upper()} with 10-fold CV on TRAINING SET ONLY...")
            
            fold_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            
            cv_mean = np.mean(fold_scores)
            cv_std = np.std(fold_scores)
            cv_min = np.min(fold_scores)
            cv_max = np.max(fold_scores)
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            test_accuracy = accuracy_score(y_test, y_pred)
            
            auc_roc = 0
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_roc = auc(fpr, tpr)
            
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
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'auc_roc': auc_roc
            }
            
            print(f"   [RESULTS] {name.upper()}")
            print(f"      10-Fold CV Mean (Train only): {cv_mean:.4f}")
            print(f"      10-Fold CV Std:  {cv_std:.4f}")
            print(f"      10-Fold Range:   [{cv_min:.4f}, {cv_max:.4f}]")
            print(f"      Test Accuracy (Unseen data): {test_accuracy:.4f}")
            print(f"      AUC-ROC:         {auc_roc:.4f}")
            
            gap = abs(test_accuracy - cv_mean)
            if gap > 0.15:
                print(f"      [WARNING] Large gap between CV and Test ({gap:.4f}) - Possible overfitting!")
        
        self._save_results_to_json(results, 'ensemble_comparison/ensemble_models_results.json')
        
        return results
    
    def _save_results_to_json(self, results, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        json_results = {}
        for name, result in results.items():
            json_results[name] = {
                'accuracy': float(result['accuracy']),
                'cv_10fold_mean': float(result['cv_10fold_mean']),
                'cv_10fold_std': float(result['cv_10fold_std']),
                'cv_10fold_min': float(result['cv_10fold_min']),
                'cv_10fold_max': float(result['cv_10fold_max']),
                'auc_roc': float(result.get('auc_roc', 0)),
                'classification_report': result['classification_report']
            }
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"[SAVED] Ensemble results saved to {filepath}")

    def visualize_10fold_results(self, results, output_dir='ensemble_comparison'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        model_names = list(results.keys())
        colors = ['#f59e0b', '#10b981', '#8b5cf6', '#ef4444']
        
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
            axes[0, 0].boxplot([model_data], positions=[idx], widths=0.6,
                                patch_artist=True, 
                                boxprops=dict(facecolor=colors[idx % len(colors)], alpha=0.7),
                                medianprops=dict(color='black', linewidth=2))
            box_positions.append(idx)
        
        axes[0, 0].set_xticks(box_positions)
        axes[0, 0].set_xticklabels([name.upper() for name in model_names], rotation=15, ha='right')
        axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('10-Fold Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0.0, 1.0])
        
        for idx, name in enumerate(model_names):
            fold_scores = results[name]['cv_10fold_scores']
            axes[0, 1].plot(range(1, 11), fold_scores, marker='o', linewidth=2, 
                          label=name.upper(), color=colors[idx % len(colors)])
        
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
                              color=[colors[i % len(colors)] for i in range(len(model_names))], alpha=0.8)
        bars2 = axes[1, 0].bar(x + width/2, test_accs, width, label='Test Accuracy', 
                              color=[colors[i % len(colors)] for i in range(len(model_names))], alpha=0.5)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        axes[1, 0].set_xlabel('Ensemble Models', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('10-Fold CV Mean vs Test Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([name.upper() for name in model_names], rotation=15, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_ylim([0.0, 1.0])
        
        cv_stds = [results[name]['cv_10fold_std'] for name in model_names]
        
        bars = axes[1, 1].bar([name.upper() for name in model_names], cv_stds, 
                             color=[colors[i % len(colors)] for i in range(len(model_names))], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        axes[1, 1].set_xlabel('Ensemble Models', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('10-Fold CV Standard Deviation (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticklabels([name.upper() for name in model_names], rotation=15, ha='right')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ensemble_10fold_cv_analysis.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_dir}/ensemble_10fold_cv_analysis.png")
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
        detailed_df.to_csv(f'{output_dir}/ensemble_10fold_detailed_scores.csv', index=False)
        print(f"[SAVED] {output_dir}/ensemble_10fold_detailed_scores.csv")

    def print_10fold_summary(self, results):
        print("\n" + "="*90)
        print("ENSEMBLE 10-FOLD CROSS-VALIDATION SUMMARY")
        print("="*90)
        
        print(f"\n{'Model':<20} {'CV Mean':<12} {'CV Std':<12} {'CV Min':<12} {'CV Max':<12} {'Test Acc':<12}")
        print("-"*90)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_10fold_mean'], reverse=True)
        
        for name, result in sorted_results:
            print(f"{name.upper():<20} "
                  f"{result['cv_10fold_mean']:<12.4f} "
                  f"{result['cv_10fold_std']:<12.4f} "
                  f"{result['cv_10fold_min']:<12.4f} "
                  f"{result['cv_10fold_max']:<12.4f} "
                  f"{result['accuracy']:<12.4f}")
        
        print("\n" + "="*90)
        print("INTERPRETATION")
        print("="*90)
        print("CV Mean: Average accuracy across 10 folds (higher is better)")
        print("CV Std:  Consistency across folds (lower is better)")
        print("CV Min:  Worst-case performance")
        print("CV Max:  Best-case performance")
        print("Test Acc: Performance on held-out test set")
        
        most_consistent = min(results.items(), key=lambda x: x[1]['cv_10fold_std'])
        print(f"\nMost Consistent: {most_consistent[0].upper()} (Std: {most_consistent[1]['cv_10fold_std']:.4f})")
        
        best_avg = max(results.items(), key=lambda x: x[1]['cv_10fold_mean'])
        print(f"Best Average: {best_avg[0].upper()} (CV Mean: {best_avg[1]['cv_10fold_mean']:.4f})")
    
    def compare_ensemble_visualization(self, results, output_dir='ensemble_comparison'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name]['cv_10fold_mean'] for name in model_names]
        
        x_pos = np.arange(len(model_names))
        
        axes[0, 0].bar(x_pos, accuracies, alpha=0.8, color='#f59e0b', label='Test Accuracy')
        axes[0, 0].bar(x_pos + 0.35, cv_means, alpha=0.8, color='#10b981', width=0.35, label='10-Fold CV Mean')
        axes[0, 0].set_xlabel('Ensemble Models', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Ensemble Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x_pos + 0.175)
        axes[0, 0].set_xticklabels([name.upper() for name in model_names], rotation=15, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
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
        axes[0, 1].set_xticklabels([name.upper() for name in model_names], rotation=15, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        for name in model_names:
            if results[name]['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, results[name]['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                axes[1, 0].plot(fpr, tpr, label=f'{name.upper()} (AUC={roc_auc:.3f})', linewidth=2)
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        axes[1, 0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].legend(loc='lower right', fontsize=8)
        axes[1, 0].grid(alpha=0.3)
        
        cv_data = []
        for name in model_names:
            cv_data.append({
                'Model': name.upper(),
                'CV Mean': results[name]['cv_10fold_mean'],
                'CV Std': results[name]['cv_10fold_std']
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
        print(f"[SAVED] {output_dir}/ensemble_comparison_dashboard.png")
        plt.close()
        
        num_models = len(model_names)
        cols = 2
        rows = (num_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
        if num_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, name in enumerate(model_names):
            cm = np.array(results[name]['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[idx],
                    xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
            axes[idx].set_title(f'{name.upper()}\nAccuracy: {results[name]["accuracy"]:.3f}', 
                            fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        for idx in range(num_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ensemble_confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_dir}/ensemble_confusion_matrices.png")
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
        summary_df.to_csv(f'{output_dir}/ensemble_performance_summary.csv', index=False)
        print(f"[SAVED] {output_dir}/ensemble_performance_summary.csv")
        
        return summary_df
    
    def compare_base_vs_ensemble_models(self, ensemble_results, base_results_path='model_comparison/base_models_results.json', output_dir='ensemble_comparison'):
        """Compare base models vs ensemble models across all metrics"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("\n" + "="*90)
        print("BASE MODELS vs ENSEMBLE MODELS COMPARISON")
        print("="*90)
        
        base_results = {}
        if os.path.exists(base_results_path):
            import json
            with open(base_results_path, 'r') as f:
                base_results = json.load(f)
            print(f"[INFO] Loaded {len(base_results)} base model results")
        else:
            print(f"[WARNING] Base model results not found at {base_results_path}")
            return None
        
        comparison_data = []
        
        for name, result in base_results.items():
            if isinstance(result, dict):
                comparison_data.append({
                    'Type': 'Base',
                    'Model': name.upper(),
                    'Accuracy': result.get('accuracy', 0),
                    'Precision': result.get('classification_report', {}).get('1', {}).get('precision', 0),
                    'Recall': result.get('classification_report', {}).get('1', {}).get('recall', 0),
                    'F1-Score': result.get('classification_report', {}).get('1', {}).get('f1-score', 0),
                    'AUC-ROC': result.get('auc_roc', 0),
                    'CV Mean': result.get('cv_10fold_mean', 0),
                    'CV Std': result.get('cv_10fold_std', 0)
                })
        
        for name, result in ensemble_results.items():
            comparison_data.append({
                'Type': 'Ensemble',
                'Model': name.upper(),
                'Accuracy': result['accuracy'],
                'Precision': result['classification_report']['1']['precision'],
                'Recall': result['classification_report']['1']['recall'],
                'F1-Score': result['classification_report']['1']['f1-score'],
                'AUC-ROC': result.get('auc_roc', 0),
                'CV Mean': result['cv_10fold_mean'],
                'CV Std': result['cv_10fold_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        csv_path = f'{output_dir}/base_vs_ensemble_comparison.csv'
        comparison_df.to_csv(csv_path, index=False)
        print(f"[SAVED] {csv_path}")
        
        print(f"\n{'Type':<10} {'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
        print("-"*90)
        for _, row in comparison_df.iterrows():
            print(f"{row['Type']:<10} {row['Model']:<25} {row['Accuracy']:<10.4f} {row['Precision']:<10.4f} "
                f"{row['Recall']:<10.4f} {row['F1-Score']:<10.4f} {row['AUC-ROC']:<10.4f}")
        
        self._visualize_base_vs_ensemble(comparison_df, output_dir)
        
        return comparison_df
    
    def create_ensemble_prediction_comparison(self, results, output_dir='ensemble_comparison'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        num_models = len(results)
        cols = 2
        rows = (num_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
        if num_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(results.items()):
            y_pred = result['y_pred']
            
            comparison_df = pd.DataFrame({
                'Index': range(len(self.y_test)),
                'Actual': self.y_test,
                'Predicted': y_pred
            })
            
            axes[idx].scatter(comparison_df['Index'], comparison_df['Actual'], 
                            alpha=0.6, label='Actual', color='blue', s=50)
            axes[idx].scatter(comparison_df['Index'], comparison_df['Predicted'], 
                            alpha=0.6, label='Predicted', color='orange', s=30, marker='x')
            
            incorrect_mask = self.y_test != y_pred
            if incorrect_mask.sum() > 0:
                axes[idx].scatter(comparison_df[incorrect_mask]['Index'], 
                                comparison_df[incorrect_mask]['Actual'],
                                color='red', s=100, marker='o', 
                                facecolors='none', linewidths=2, 
                                label='Incorrect')
            
            axes[idx].set_xlabel('Test Sample Index', fontsize=10)
            axes[idx].set_ylabel('Class (0=Fail, 1=Pass)', fontsize=10)
            axes[idx].set_title(f'{model_name.upper()}\nAccuracy: {result["accuracy"]:.3f}', 
                            fontsize=12, fontweight='bold')
            axes[idx].legend(loc='upper right')
            axes[idx].grid(alpha=0.3)
            axes[idx].set_ylim(-0.2, 1.2)
        
        for idx in range(num_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ensemble_prediction_comparison.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_dir}/ensemble_prediction_comparison.png")
        plt.close()

    def _visualize_base_vs_ensemble(self, comparison_df, output_dir):
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'CV Mean']
        positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
        
        base_color = '#3498db'
        ensemble_color = '#f59e0b'
        
        for metric, (row, col) in zip(metrics, positions):
            ax = axes[row, col]
            
            base_data = comparison_df[comparison_df['Type'] == 'Base'].sort_values('Model')
            ensemble_data = comparison_df[comparison_df['Type'] == 'Ensemble'].sort_values('Model')
            
            num_base = len(base_data)
            num_ensemble = len(ensemble_data)
            
            x_base = np.arange(num_base)
            x_ensemble = np.arange(num_base, num_base + num_ensemble)
            
            width = 0.7
            
            bars1 = ax.bar(x_base, base_data[metric], width, 
                        label='Base Models', color=base_color, alpha=0.8)
            bars2 = ax.bar(x_ensemble, ensemble_data[metric], width,
                        label='Ensemble Models', color=ensemble_color, alpha=0.8)
            
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            all_models = list(base_data['Model']) + list(ensemble_data['Model'])
            ax.set_xticks(list(x_base) + list(x_ensemble))
            ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.0])
            
            if num_base > 0 and num_ensemble > 0:
                ax.axvline(x=num_base - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/base_vs_ensemble_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_dir}/base_vs_ensemble_metrics_comparison.png")
        plt.close()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        base_data = comparison_df[comparison_df['Type'] == 'Base']
        ensemble_data = comparison_df[comparison_df['Type'] == 'Ensemble']
        
        base_avg = base_data[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].mean()
        ensemble_avg = ensemble_data[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].mean()
        
        x = np.arange(len(base_avg))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, base_avg.values, width, 
                        label='Base Models Avg', color=base_color, alpha=0.8)
        bars2 = axes[0].bar(x + width/2, ensemble_avg.values, width,
                        label='Ensemble Models Avg', color=ensemble_color, alpha=0.8)
        
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(base_avg.index, fontsize=11)
        axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Average Performance: Base vs Ensemble', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1.0])
        
        from math import pi
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        ax = plt.subplot(122, projection='polar')
        
        base_values = list(base_avg.values) + [base_avg.values[0]]
        ensemble_values = list(ensemble_avg.values) + [ensemble_avg.values[0]]
        
        ax.plot(angles, base_values, 'o-', linewidth=2, 
            label='Base Models Avg', color=base_color)
        ax.fill(angles, base_values, alpha=0.25, color=base_color)
        
        ax.plot(angles, ensemble_values, 'o-', linewidth=2,
            label='Ensemble Models Avg', color=ensemble_color)
        ax.fill(angles, ensemble_values, alpha=0.25, color=ensemble_color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar: Base vs Ensemble', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/base_vs_ensemble_summary.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_dir}/base_vs_ensemble_summary.png")
        plt.close()
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        base_data_sorted = comparison_df[comparison_df['Type'] == 'Base'].sort_values('Model')
        ensemble_data_sorted = comparison_df[comparison_df['Type'] == 'Ensemble'].sort_values('Model')
        
        sorted_df = pd.concat([base_data_sorted, ensemble_data_sorted])
        
        x = np.arange(len(sorted_df))
        width = 0.15
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        colors_grouped = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        
        for i, metric in enumerate(metrics):
            offset = width * (i - len(metrics)/2 + 0.5)
            bars = ax.bar(x + offset, sorted_df[metric], width, 
                        label=metric, color=colors_grouped[i], alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('All Metrics Comparison: Base vs Ensemble Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_df['Model'].values, 
                        rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        num_base = len(base_data_sorted)
        if num_base > 0:
            ax.axvline(x=num_base - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
            
            ax.text(num_base/2 - 0.5, 0.95, 
                'Base Models', ha='center', fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor=base_color, alpha=0.3))
            
            ax.text(num_base + len(ensemble_data_sorted)/2 - 0.5, 0.95,
                'Ensemble Models', ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=ensemble_color, alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/base_vs_ensemble_all_metrics.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_dir}/base_vs_ensemble_all_metrics.png")
        plt.close()
        
        print("\n[STATISTICS] Performance Statistics:")
        print("\nBase Models:")
        base_stats = comparison_df[comparison_df['Type'] == 'Base'][['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].describe()
        print(base_stats)
        
        print("\nEnsemble Models:")
        ensemble_stats = comparison_df[comparison_df['Type'] == 'Ensemble'][['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].describe()
        print(ensemble_stats)
        
        print("\n[BEST PERFORMERS]")
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
            best = comparison_df.loc[comparison_df[metric].idxmax()]
            print(f"Best {metric}: {best['Model']} ({best['Type']}) = {best[metric]:.4f}")
    
    def create_ensemble_accuracy_comparison(self, results, output_dir='ensemble_comparison'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name]['cv_10fold_mean'] for name in model_names]
        
        colors = ['#f59e0b', '#10b981', '#8b5cf6', '#ef4444']
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, accuracies, width, label='Test Accuracy', color=[colors[i % len(colors)] for i in range(len(model_names))], alpha=0.8)
        bars2 = axes[0].bar(x + width/2, cv_means, width, label='10-Fold CV Mean', color=[colors[i % len(colors)] for i in range(len(model_names))], alpha=0.5)
        
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[0].set_xlabel('Ensemble Models', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_title('Ensemble Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([name.upper() for name in model_names], rotation=15, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1.0])
        
        cv_stds = [results[name]['cv_10fold_std'] for name in model_names]
        
        axes[1].bar([name.upper() for name in model_names], accuracies, yerr=cv_stds, 
                capsize=10, color=[colors[i % len(colors)] for i in range(len(model_names))], alpha=0.8, ecolor='black', linewidth=2)
        
        for i, (acc, std) in enumerate(zip(accuracies, cv_stds)):
            axes[1].text(i, acc + std + 0.02, f'{acc:.3f}\n±{std:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
        
        axes[1].set_xlabel('Ensemble Models', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title('Accuracy with Standard Deviation', fontsize=14, fontweight='bold')
        axes[1].set_xticklabels([name.upper() for name in model_names], rotation=15, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim([0, 1.0])
        
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
                label=name.upper(), color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Comparison', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ensemble_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_dir}/ensemble_accuracy_comparison.png")
        plt.close()
        
        print(f"\n{'='*90}")
        print(f"4 ENSEMBLE MODELS - DETAILED ACCURACY COMPARISON")
        print(f"{'='*90}")
        print(f"\n{'Rank':<5} {'Model':<25} {'Test Acc':<12} {'CV Mean':<12} {'CV Std':<10} {'F1-Score':<10}")
        print(f"{'-'*90}")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            f1 = result['classification_report']['1']['f1-score']
            print(f"{rank:<5} {name.upper():<25} {result['accuracy']:<12.4f} "
                f"{result['cv_10fold_mean']:<12.4f} {result['cv_10fold_std']:<10.4f} {f1:<10.4f}")
        
        return results

    def evaluate_ensemble_predictions(self, results, output_dir='ensemble_comparison'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("\n" + "="*60)
        print("[EVALUATION] DETAILED ENSEMBLE PREDICTIONS ANALYSIS")
        print("="*60)
        
        for model_name, result in results.items():
            y_pred = result['y_pred']
            
            prediction_details = pd.DataFrame({
                'Actual': self.y_test,
                'Predicted': y_pred,
                'Correct': self.y_test == y_pred,
                'Actual_Label': ['Pass' if y == 1 else 'Fail' for y in self.y_test],
                'Predicted_Label': ['Pass' if y == 1 else 'Fail' for y in y_pred]
            })
            
            if result['y_pred_proba'] is not None:
                prediction_details['Pass_Probability'] = result['y_pred_proba']
            
            total_samples = len(self.y_test)
            correct_predictions = (self.y_test == y_pred).sum()
            incorrect_predictions = total_samples - correct_predictions
            
            print(f"\n[MODEL] {model_name.upper()}")
            print(f"   Total Test Samples: {total_samples}")
            print(f"   Correct Predictions: {correct_predictions} ({correct_predictions/total_samples*100:.2f}%)")
            print(f"   Incorrect Predictions: {incorrect_predictions} ({incorrect_predictions/total_samples*100:.2f}%)")
            
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            print(f"\n   [BREAKDOWN]")
            print(f"   True Positives (Predicted Pass, Actually Pass): {tp}")
            print(f"   True Negatives (Predicted Fail, Actually Fail): {tn}")
            print(f"   False Positives (Predicted Pass, Actually Fail): {fp}")
            print(f"   False Negatives (Predicted Fail, Actually Pass): {fn}")
            
            csv_file = f'{output_dir}/{model_name}_test_predictions.csv'
            prediction_details.to_csv(csv_file, index=True)
            print(f"\n   [SAVED] {csv_file}")
    


    def generate_ensemble_report(self, results, output_dir='ensemble_comparison'):
        report_path = f'{output_dir}/ensemble_evaluation_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Ensemble Models Testing Report (10-Fold Cross-Validation)\n\n")
            f.write(f"## Test Dataset Information\n\n")
            f.write(f"Total Test Samples: {len(self.y_test)}\n")
            f.write(f"Pass Count: {(self.y_test == 1).sum()}\n")
            f.write(f"Fail Count: {(self.y_test == 0).sum()}\n")
            f.write(f"Pass Rate: {(self.y_test == 1).mean():.2%}\n\n")
            
            f.write("## Ensemble Performance on Test Data\n\n")
            
            for model_name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
                y_pred = result['y_pred']
                cm = confusion_matrix(self.y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                f.write(f"### {model_name.upper()}\n\n")
                f.write(f"**Overall Accuracy:** {result['accuracy']:.4f}\n\n")
                
                f.write(f"**10-Fold Cross-Validation:**\n\n")
                f.write(f"- **CV Mean:** {result['cv_10fold_mean']:.4f}\n")
                f.write(f"- **CV Std:** {result['cv_10fold_std']:.4f}\n")
                f.write(f"- **CV Min:** {result['cv_10fold_min']:.4f}\n")
                f.write(f"- **CV Max:** {result['cv_10fold_max']:.4f}\n\n")
                
                f.write(f"**Confusion Matrix:**\n\n")
                f.write(f"```\n")
                f.write(f"                Predicted Fail    Predicted Pass\n")
                f.write(f"Actual Fail          {tn:4d}              {fp:4d}\n")
                f.write(f"Actual Pass          {fn:4d}              {tp:4d}\n")
                f.write(f"```\n\n")
                
                f.write(f"**Detailed Metrics:**\n\n")
                f.write(f"- **True Positives:** {tp} ({tp/len(self.y_test)*100:.1f}%)\n")
                f.write(f"- **True Negatives:** {tn} ({tn/len(self.y_test)*100:.1f}%)\n")
                f.write(f"- **False Positives:** {fp} ({fp/len(self.y_test)*100:.1f}%)\n")
                f.write(f"- **False Negatives:** {fn} ({fn/len(self.y_test)*100:.1f}%)\n\n")
                
                report = result['classification_report']
                f.write(f"**Classification Report:**\n\n")
                f.write(f"- **Precision (Pass class):** {report['1']['precision']:.4f}\n")
                f.write(f"- **Recall (Pass class):** {report['1']['recall']:.4f}\n")
                f.write(f"- **F1-Score (Pass class):** {report['1']['f1-score']:.4f}\n\n")
                f.write("---\n\n")
            
            f.write("## Ensemble Methods Explained\n\n")
            f.write("**Bagging:** 10 Random Forest models with bootstrap sampling\n\n")
            f.write("**Boosting:** Gradient Boosting with 100 estimators\n\n")
            f.write("**Stacking:** Combines 6 fresh base model instances (KNN, Decision Tree, Random Forest, SVM, Neural Network, Naive Bayes) with Logistic Regression meta-learner\n\n")
            
            f.write("## 10-Fold Cross-Validation\n\n")
            f.write("Cross-validation performed on training set only to prevent data leakage.\n")
            f.write("Test set kept completely separate for final evaluation.\n")
        
        print(f"[SAVED] {report_path}")
        return report_path
    
    def predict_with_ensembles(self, X):
        predictions = {}
        
        for name, model in self.ensemble_models.items():
            pred = model.predict(X)
            pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            
            predictions[name] = {
                'prediction': pred,
                'probability': pred_proba
            }
        
        return predictions
    
    def save_ensemble_models(self, directory='saved_classification_ensemble_models'):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for name, model in self.ensemble_models.items():
            joblib.dump(model, os.path.join(directory, f'{name}_ensemble.pkl'))
        
        print(f"[SAVE] Ensemble models saved to {directory}/")

    def load_ensemble_models(self, directory='saved_classification_ensemble_models'):
        try:
            ensemble_files = [f for f in os.listdir(directory) if f.endswith('_ensemble.pkl')]
            
            for file in ensemble_files:
                model_name = file.replace('_ensemble.pkl', '')
                model_path = os.path.join(directory, file)
                self.ensemble_models[model_name] = joblib.load(model_path)
            
            print(f"[LOAD] Loaded {len(self.ensemble_models)} ensemble models from {directory}/")
            return True
            
        except Exception as e:
            print(f"[ERROR] {e}")
            return False

def main():
    from base_models import SocialWorkPredictorModels
    
    print("="*60)
    print("[START] TRAINING ENSEMBLE MODELS WITH 10-FOLD CV")
    print("="*60)
    
    predictor = SocialWorkPredictorModels()
    
    data = predictor.load_preprocessed_data(data_dir='classification_processed_data', approach='label')
    
    if data is None:
        print("\n[ERROR] Could not load preprocessed data")
        return
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print("\n[INFO] Loading base models to extract best hyperparameters...")
    if not predictor.load_models('saved_models'):
        print("[WARNING] Could not load base models, using default hyperparameters")
        best_params = {}
    else:
        best_params = {}
        results_file = 'model_comparison/base_models_results.json'
        if os.path.exists(results_file):
            import json
            with open(results_file, 'r') as f:
                results_data = json.load(f)
                for model_name, model_data in results_data.items():
                    if 'best_params' in model_data:
                        best_params[model_name] = model_data['best_params']
            print(f"[INFO] Loaded hyperparameters for {len(best_params)} base models")
        else:
            print("[WARNING] No results file found, using default hyperparameters")
    
    ensemble = EnsembleModels(best_params_dict=best_params)
    ensemble_results = ensemble.train_ensemble_models_with_10fold(X_train, y_train, X_test, y_test)
    
    ensemble.print_10fold_summary(ensemble_results)
    
    print(f"\n[VISUALIZATION] Creating 10-fold CV analysis...")
    ensemble.visualize_10fold_results(ensemble_results)
    
    print("\n" + "="*60)
    print("[RESULTS] ENSEMBLE MODEL PERFORMANCE")
    print("="*60)
    
    sorted_results = sorted(ensemble_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Model':<20} {'Accuracy':<12} {'CV Score':<15}")
    print("-" * 60)
    for i, (model_name, metrics) in enumerate(sorted_results, 1):
        cv_info = f"{metrics['cv_10fold_mean']:.4f}±{metrics['cv_10fold_std']:.3f}"
        print(f"{i:<5} {model_name.upper():<20} {metrics['accuracy']:<12.4f} {cv_info:<15}")
    
    ensemble.evaluate_ensemble_predictions(ensemble_results)
    
    print(f"\n[VISUALIZATION] Creating comparison graphs...")
    summary_df = ensemble.compare_ensemble_visualization(ensemble_results)
    
    ensemble.create_ensemble_accuracy_comparison(ensemble_results)
    
    ensemble.create_ensemble_prediction_comparison(ensemble_results)
    
    ensemble.generate_ensemble_report(ensemble_results)
    
    print(f"\n[COMPARISON] Comparing Base vs Ensemble Models...")
    comparison_df = ensemble.compare_base_vs_ensemble_models(ensemble_results)
    
    print(f"\n[SUMMARY] Performance Summary:")
    print(summary_df.to_string(index=False))
    
    if comparison_df is not None:
        print(f"\n[COMPARISON SUMMARY]")
        print(comparison_df.to_string(index=False))
    
    ensemble.save_ensemble_models()
    
    print(f"\n[COMPLETE] Training completed successfully")
    print(f"[OUTPUT] Generated files:")
    print(f"   - ensemble_comparison/ensemble_10fold_cv_analysis.png")
    print(f"   - ensemble_comparison/ensemble_comparison_dashboard.png")
    print(f"   - ensemble_comparison/ensemble_confusion_matrices.png")
    print(f"   - ensemble_comparison/ensemble_accuracy_comparison.png")
    print(f"   - ensemble_comparison/ensemble_prediction_comparison.png")
    print(f"   - ensemble_comparison/ensemble_evaluation_report.md")
    print(f"   - ensemble_comparison/base_vs_ensemble_comparison.csv")
    print(f"   - ensemble_comparison/base_vs_ensemble_metrics_comparison.png")
    print(f"   - ensemble_comparison/base_vs_ensemble_summary.png")
    print(f"   - ensemble_comparison/base_vs_ensemble_all_metrics.png")

if __name__ == "__main__":
    main()