import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, pointbiserialr, chi2_contingency, f_oneway
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class CorrelationAnalyzer:
    def __init__(self):
        # Categorical columns (non-numeric, multi-class)
        self.categorical_columns = ['Gender', 'IncomeLevel', 'EmploymentStatus']
        # Numerical columns including ExamResultPercent
        self.numerical_columns = [
            'Age', 'StudyHours', 'SleepHours', 'Confidence',
            'MockExamScore', 'GPA', 'InternshipGrade', 'TestAnxiety',
            'SocialSupport', 'MotivationScore', 'EnglishProficiency',
            'ExamResultPercent'
        ]
        # Binary columns (0/1 values)
        self.binary_columns = ['ReviewCenter', 'Scholarship']
        # Target column for classification
        self.target_column = 'Passed'
        self.correlation_matrix = None
        self.correlation_results = {}

    def load_data(self, file_path):
        """Load CSV data and print basic info."""
        try:
            df = pd.read_csv(file_path)
            print(f"[SUCCESS] Data loaded. Shape: {df.shape}")
            print(f"[INFO] Columns found: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"[ERROR] Error loading file: {e}")
            return None

    def prepare_encoded_data(self, df):
        """Encode categorical columns using LabelEncoder."""
        df_encoded = df.copy()
        le = LabelEncoder()

        for col in self.categorical_columns:
            if col in df_encoded.columns:
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        return df_encoded

    def compute_correlation_matrix(self, df):
        """Compute Pearson correlation matrix for continuous and binary features."""
        print("\n" + "="*70)
        print("CORRELATION MATRIX ANALYSIS (CONTINUOUS + BINARY VARIABLES)")
        print("="*70)

        continuous_features = self.numerical_columns + self.binary_columns
        available_continuous = [col for col in continuous_features if col in df.columns]

        # Include target column in the matrix
        if self.target_column in df.columns:
            available_continuous.append(self.target_column)

        print(f"[INFO] Features in correlation matrix: {available_continuous}")

        correlation_matrix = df[available_continuous].corr()
        self.correlation_matrix = correlation_matrix

        # Show correlations with target
        if self.target_column in correlation_matrix.columns:
            target_corr = correlation_matrix[self.target_column].drop(self.target_column).abs().sort_values(ascending=False)
            print(f"\n[TARGET CORRELATION] Continuous features correlated with {self.target_column}:")
            for feature, corr in target_corr.items():
                print(f"   {feature}: {corr:.4f}")

        # Detect high inter-feature correlations (excluding target)
        print(f"\n[INTER-FEATURE] High inter-feature correlations (>0.5):")
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                col_i = correlation_matrix.columns[i]
                col_j = correlation_matrix.columns[j]
                if corr_val > 0.5 and col_i != self.target_column and col_j != self.target_column:
                    high_corr_pairs.append((col_i, col_j, corr_val))

        if high_corr_pairs:
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
                print(f"   {feat1} <-> {feat2}: {corr:.4f}")
        else:
            print("   [SUCCESS] No high inter-feature correlations found")

        # Store results
        self.correlation_results['correlation_matrix'] = correlation_matrix.to_dict()
        self.correlation_results['high_correlations'] = [
            {'feature1': feat1, 'feature2': feat2, 'correlation': float(corr)}
            for feat1, feat2, corr in high_corr_pairs
        ]

        return correlation_matrix

    def pearson_correlation_analysis(self, df):
        """Compute pairwise Pearson correlation for all continuous and binary features."""
        print("\n" + "="*70)
        print("PEARSON CORRELATION (CONTINUOUS -> CONTINUOUS)")
        print("="*70)

        continuous_features = self.numerical_columns + self.binary_columns
        available_features = [col for col in continuous_features if col in df.columns]

        pearson_correlations = {}

        for col in available_features:
            try:
                col_data = df[col].dropna()
                for other_col in available_features:
                    if col != other_col and other_col in df.columns:
                        other_data = df[other_col].dropna()
                        common_idx = col_data.index.intersection(other_data.index)

                        if len(common_idx) > 0:
                            corr_coef, p_value = pearsonr(col_data.loc[common_idx], other_data.loc[common_idx])

                            pair_key = f"{col}_vs_{other_col}"
                            reverse_key = f"{other_col}_vs_{col}"
                            # Avoid duplicate pairs
                            if pair_key not in pearson_correlations and reverse_key not in pearson_correlations:
                                pearson_correlations[pair_key] = {
                                    'feature1': col,
                                    'feature2': other_col,
                                    'correlation': float(corr_coef),
                                    'p_value': float(p_value),
                                    'significant': bool(p_value < 0.05)
                                }
            except Exception as e:
                print(f"   {col}: Failed ({e})")

        self.correlation_results['pearson_correlation'] = pearson_correlations
        print(f"[SUCCESS] Computed {len(pearson_correlations)} pairwise Pearson correlations")

        return pearson_correlations

    def point_biserial_correlation_analysis(self, df):
        """Compute point-biserial correlation between continuous features and binary target."""
        print("\n" + "="*70)
        print("POINT-BISERIAL CORRELATION (CONTINUOUS -> BINARY TARGET)")
        print("="*70)

        if self.target_column not in df.columns:
            print("[ERROR] Target column not found")
            return {}

        continuous_features = self.numerical_columns + self.binary_columns
        available_features = [col for col in continuous_features if col in df.columns]

        point_biserial_correlations = {}

        for col in available_features:
            try:
                col_data = df[col].dropna()
                target_data = df[self.target_column].loc[col_data.index]

                corr_coef, p_value = pointbiserialr(target_data, col_data)
                point_biserial_correlations[col] = {
                    'correlation': float(corr_coef),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }
                sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"   {col}: rpb={corr_coef:.4f}, p={p_value:.4f} {sig_marker}")
            except Exception as e:
                print(f"   {col}: Failed ({e})")

        self.correlation_results['point_biserial_correlation'] = point_biserial_correlations

        return point_biserial_correlations

    def chi_square_and_cramers_v_analysis(self, df):
        """Compute Chi-Square and Cramer's V between categorical features and binary target."""
        print("\n" + "="*70)
        print("CHI-SQUARE TEST + CRAMER'S V (CATEGORICAL -> BINARY TARGET)")
        print("="*70)

        if self.target_column not in df.columns:
            print("[ERROR] Target column not found")
            return {}

        chi_square_results = {}

        for col in self.categorical_columns:
            if col in df.columns:
                try:
                    contingency_table = pd.crosstab(df[col], df[self.target_column])
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                    n = contingency_table.sum().sum()
                    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
                    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

                    chi_square_results[col] = {
                        'chi2': float(chi2),
                        'p_value': float(p_value),
                        'dof': int(dof),
                        'cramers_v': float(cramers_v),
                        'significant': bool(p_value < 0.05)
                    }

                    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"   {col}: chi2={chi2:.4f}, p={p_value:.4f}, Cramer's V={cramers_v:.4f} {sig_marker}")
                except Exception as e:
                    print(f"   {col}: Failed ({e})")

        self.correlation_results['chi_square_cramers_v'] = chi_square_results

        return chi_square_results

    def cramers_v_categorical_analysis(self, df):
        """Compute Cramer's V between all pairs of categorical features."""
        print("\n" + "="*70)
        print("CRAMER'S V (CATEGORICAL -> CATEGORICAL)")
        print("="*70)

        cramers_v_results = {}

        for i, col1 in enumerate(self.categorical_columns):
            if col1 in df.columns:
                for col2 in self.categorical_columns[i+1:]:
                    if col2 in df.columns:
                        try:
                            contingency_table = pd.crosstab(df[col1], df[col2])
                            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                            n = contingency_table.sum().sum()
                            min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
                            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

                            pair_key = f"{col1}_vs_{col2}"
                            cramers_v_results[pair_key] = {
                                'feature1': col1,
                                'feature2': col2,
                                'chi2': float(chi2),
                                'p_value': float(p_value),
                                'cramers_v': float(cramers_v),
                                'significant': bool(p_value < 0.05)
                            }

                            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                            print(f"   {col1} vs {col2}: Cramer's V={cramers_v:.4f}, p={p_value:.4f} {sig_marker}")
                        except Exception as e:
                            print(f"   {col1} vs {col2}: Failed ({e})")

        self.correlation_results['cramers_v_categorical'] = cramers_v_results

        return cramers_v_results

    def one_way_anova_analysis(self, df):
        """Compute One-Way ANOVA between categorical and continuous features."""
        print("\n" + "="*70)
        print("ONE-WAY ANOVA (CATEGORICAL -> CONTINUOUS)")
        print("="*70)

        anova_results = {}

        for cat_col in self.categorical_columns:
            if cat_col not in df.columns:
                continue

            for num_col in self.numerical_columns:
                if num_col not in df.columns:
                    continue

                try:
                    groups = []
                    categories = df[cat_col].dropna().unique()

                    for category in categories:
                        group_data = df[df[cat_col] == category][num_col].dropna().values
                        if len(group_data) > 0:
                            groups.append(group_data)

                    if len(groups) >= 2:
                        f_stat, p_value = f_oneway(*groups)

                        pair_key = f"{cat_col}_vs_{num_col}"
                        anova_results[pair_key] = {
                            'categorical_feature': cat_col,
                            'numerical_feature': num_col,
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': bool(p_value < 0.05),
                            'n_groups': len(groups)
                        }

                        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                        print(f"   {cat_col} -> {num_col}: F={f_stat:.4f}, p={p_value:.4f} {sig_marker}")

                except Exception as e:
                    print(f"   {cat_col} -> {num_col}: Failed ({e})")

        self.correlation_results['one_way_anova'] = anova_results
        print(f"[SUCCESS] Completed {len(anova_results)} ANOVA tests")

        return anova_results

    def visualize_correlation_matrix(self, output_dir='correlation_analysis'):
        """Generate and save the correlation matrix heatmap."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.correlation_matrix is None:
            print("[ERROR] No correlation matrix computed")
            return

        plt.figure(figsize=(16, 14))
        sns.heatmap(self.correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Continuous + Binary Features Correlation Matrix (Pearson r)', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Correlation matrix heatmap: {output_path}")
        plt.close()

    def visualize_target_correlations(self, output_dir='correlation_analysis'):
        """Generate and save the point-biserial correlation bar chart."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if 'point_biserial_correlation' not in self.correlation_results:
            print("[ERROR] No point-biserial correlation results")
            return

        pb_data = self.correlation_results['point_biserial_correlation']
        features = list(pb_data.keys())
        correlations = [pb_data[feat]['correlation'] for feat in features]
        p_values = [pb_data[feat]['p_value'] for feat in features]

        sorted_indices = np.argsort([abs(c) for c in correlations])[::-1]
        features_sorted = [features[i] for i in sorted_indices]
        correlations_sorted = [correlations[i] for i in sorted_indices]
        p_values_sorted = [p_values[i] for i in sorted_indices]

        colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in p_values_sorted]

        plt.figure(figsize=(12, 9))
        bars = plt.barh(features_sorted, correlations_sorted, color=colors)
        plt.xlabel('Point-Biserial Correlation Coefficient', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title(f'Continuous Features Correlations with {self.target_column}', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3)

        plt.legend([plt.Rectangle((0,0),1,1, color='#2ecc71'),
                   plt.Rectangle((0,0),1,1, color='#95a5a6')],
                  ['Significant (p<0.05)', 'Not Significant'],
                  loc='lower right')

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'target_correlations.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Target correlations plot: {output_path}")
        plt.close()

    def visualize_categorical_associations(self, output_dir='correlation_analysis'):
        """Generate and save the Cramer's V categorical associations bar chart."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if 'chi_square_cramers_v' not in self.correlation_results:
            print("[ERROR] No chi-square/Cramer's V results")
            return

        chi_data = self.correlation_results['chi_square_cramers_v']
        features = list(chi_data.keys())
        cramers_v_values = [chi_data[feat]['cramers_v'] for feat in features]
        p_values = [chi_data[feat]['p_value'] for feat in features]

        sorted_indices = np.argsort(cramers_v_values)[::-1]
        features_sorted = [features[i] for i in sorted_indices]
        cramers_v_sorted = [cramers_v_values[i] for i in sorted_indices]
        p_values_sorted = [p_values[i] for i in sorted_indices]

        colors = ['#3498db' if p < 0.05 else '#95a5a6' for p in p_values_sorted]

        plt.figure(figsize=(10, 6))
        plt.barh(features_sorted, cramers_v_sorted, color=colors)
        plt.xlabel("Cramer's V", fontsize=12, fontweight='bold')
        plt.ylabel('Categorical Features', fontsize=12, fontweight='bold')
        plt.title(f"Categorical Features Association with {self.target_column} (Cramer's V)", fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        plt.legend([plt.Rectangle((0,0),1,1, color='#3498db'),
                   plt.Rectangle((0,0),1,1, color='#95a5a6')],
                  ['Significant (p<0.05)', 'Not Significant'],
                  loc='lower right')

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'categorical_associations.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Categorical associations plot: {output_path}")
        plt.close()

    def visualize_anova_results(self, output_dir='correlation_analysis'):
        """Generate and save the ANOVA results bar chart."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if 'one_way_anova' not in self.correlation_results:
            print("[ERROR] No ANOVA results")
            return

        anova_data = self.correlation_results['one_way_anova']

        pairs = []
        f_stats = []
        p_values = []

        for pair_key, data in anova_data.items():
            pairs.append(f"{data['categorical_feature']} -> {data['numerical_feature']}")
            f_stats.append(data['f_statistic'])
            p_values.append(data['p_value'])

        sorted_indices = np.argsort(f_stats)[::-1][:20]
        pairs_sorted = [pairs[i] for i in sorted_indices]
        f_stats_sorted = [f_stats[i] for i in sorted_indices]
        p_values_sorted = [p_values[i] for i in sorted_indices]

        colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in p_values_sorted]

        plt.figure(figsize=(12, 10))
        plt.barh(pairs_sorted, f_stats_sorted, color=colors)
        plt.xlabel('F-Statistic', fontsize=12, fontweight='bold')
        plt.ylabel('Categorical -> Numerical Pairs', fontsize=12, fontweight='bold')
        plt.title('One-Way ANOVA Results (Top 20 by F-Statistic)', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        plt.legend([plt.Rectangle((0,0),1,1, color='#e74c3c'),
                   plt.Rectangle((0,0),1,1, color='#95a5a6')],
                  ['Significant (p<0.05)', 'Not Significant'],
                  loc='lower right')

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'anova_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] ANOVA results plot: {output_path}")
        plt.close()

    def visualize_exam_result_correlations(self, df, output_dir='correlation_analysis'):
        """Generate scatter plots of ExamResultPercent vs top correlated features."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if 'ExamResultPercent' not in df.columns:
            print("[ERROR] ExamResultPercent column not found")
            return

        # Compute correlations of ExamResultPercent with other numerical features
        continuous_features = [col for col in self.numerical_columns if col in df.columns and col != 'ExamResultPercent']
        corr_with_exam = {}
        for col in continuous_features:
            try:
                valid = df[['ExamResultPercent', col]].dropna()
                if len(valid) > 2:
                    corr, pval = pearsonr(valid['ExamResultPercent'], valid[col])
                    corr_with_exam[col] = {'correlation': corr, 'p_value': pval}
            except Exception:
                pass

        # Sort by absolute correlation and take top 6
        sorted_features = sorted(corr_with_exam.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)[:6]

        if not sorted_features:
            print("[INFO] No valid correlations found for ExamResultPercent")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (feat, data) in enumerate(sorted_features):
            ax = axes[idx]
            valid = df[['ExamResultPercent', feat, self.target_column]].dropna()
            colors = valid[self.target_column].map({1: '#2ecc71', 0: '#e74c3c'})
            ax.scatter(valid[feat], valid['ExamResultPercent'], c=colors, alpha=0.6, edgecolors='white', linewidth=0.5)
            ax.set_xlabel(feat, fontsize=10, fontweight='bold')
            ax.set_ylabel('ExamResultPercent', fontsize=10, fontweight='bold')
            ax.set_title(f"r={data['correlation']:.3f}, p={data['p_value']:.4f}", fontsize=10)
            ax.grid(alpha=0.3)

        # Hide unused subplots if fewer than 6 features
        for idx in range(len(sorted_features), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle('ExamResultPercent vs Top Correlated Features (Green=Passed, Red=Failed)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'exam_result_correlations.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] ExamResultPercent correlation scatter plots: {output_path}")
        plt.close()

        # Store ExamResultPercent correlations in results
        self.correlation_results['exam_result_correlations'] = {
            feat: {'correlation': float(data['correlation']), 'p_value': float(data['p_value'])}
            for feat, data in sorted_features
        }

    def save_results(self, output_dir='correlation_analysis'):
        """Save correlation results to JSON and CSV files."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        json_path = os.path.join(output_dir, 'correlation_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.correlation_results, f, indent=2)
        print(f"[SAVED] Correlation results: {json_path}")

        if self.correlation_matrix is not None:
            csv_path = os.path.join(output_dir, 'correlation_matrix.csv')
            self.correlation_matrix.to_csv(csv_path)
            print(f"[SAVED] Correlation matrix CSV: {csv_path}")

    def generate_report(self, output_dir='correlation_analysis'):
        """Generate a markdown report summarizing all correlation analysis results."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        report_path = os.path.join(output_dir, 'correlation_report.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Correlation Analysis Report\n\n")
            f.write("## Overview\n\n")
            f.write(f"Target Variable: **{self.target_column}** (Binary)\n\n")
            f.write("### Feature Groups\n\n")
            f.write(f"- **Numerical Features**: {', '.join(self.numerical_columns)}\n")
            f.write(f"- **Binary Features**: {', '.join(self.binary_columns)}\n")
            f.write(f"- **Categorical Features**: {', '.join(self.categorical_columns)}\n\n")
            f.write("## Statistical Methods Applied\n\n")
            f.write("1. **Pearson Correlation (r)**: Continuous -> Continuous\n")
            f.write("2. **Point-Biserial Correlation (rpb)**: Continuous -> Binary Target\n")
            f.write("3. **Chi-Square Test + Cramer's V**: Categorical -> Binary Target\n")
            f.write("4. **Cramer's V**: Categorical -> Categorical\n")
            f.write("5. **One-Way ANOVA**: Categorical -> Continuous\n\n")

            # Point-Biserial section
            if 'point_biserial_correlation' in self.correlation_results:
                f.write("## Point-Biserial Correlation (Continuous -> Binary Target)\n\n")
                pb_data = self.correlation_results['point_biserial_correlation']
                sorted_features = sorted(pb_data.items(),
                                       key=lambda x: abs(x[1]['correlation']),
                                       reverse=True)

                f.write("| Feature | Correlation (rpb) | P-Value | Significant |\n")
                f.write("|---------|-------------------|---------|-------------|\n")

                for feature, data in sorted_features:
                    sig = "Yes" if data['significant'] else "No"
                    f.write(f"| {feature} | {data['correlation']:.4f} | {data['p_value']:.4f} | {sig} |\n")

                f.write("\n")

            # Chi-Square section
            if 'chi_square_cramers_v' in self.correlation_results:
                f.write("## Chi-Square Test + Cramer's V (Categorical -> Binary Target)\n\n")
                chi_data = self.correlation_results['chi_square_cramers_v']
                sorted_features = sorted(chi_data.items(),
                                       key=lambda x: x[1]['cramers_v'],
                                       reverse=True)

                f.write("| Feature | Chi-Square | P-Value | Cramer's V | Significant |\n")
                f.write("|---------|------------|---------|------------|-------------|\n")

                for feature, data in sorted_features:
                    sig = "Yes" if data['significant'] else "No"
                    f.write(f"| {feature} | {data['chi2']:.4f} | {data['p_value']:.4f} | {data['cramers_v']:.4f} | {sig} |\n")

                f.write("\n")

            # Cramer's V categorical section
            if 'cramers_v_categorical' in self.correlation_results:
                cramers_data = self.correlation_results['cramers_v_categorical']
                if cramers_data:
                    f.write("## Cramer's V (Categorical -> Categorical)\n\n")
                    f.write("| Feature 1 | Feature 2 | Cramer's V | P-Value | Significant |\n")
                    f.write("|-----------|-----------|------------|---------|-------------|\n")

                    for pair_key, data in sorted(cramers_data.items(), key=lambda x: x[1]['cramers_v'], reverse=True):
                        sig = "Yes" if data['significant'] else "No"
                        f.write(f"| {data['feature1']} | {data['feature2']} | {data['cramers_v']:.4f} | {data['p_value']:.4f} | {sig} |\n")
                    f.write("\n")

            # ANOVA section
            if 'one_way_anova' in self.correlation_results:
                f.write("## One-Way ANOVA (Categorical -> Continuous)\n\n")
                anova_data = self.correlation_results['one_way_anova']
                sorted_anova = sorted(anova_data.items(),
                                    key=lambda x: x[1]['f_statistic'],
                                    reverse=True)

                f.write("| Categorical Feature | Numerical Feature | F-Statistic | P-Value | Significant |\n")
                f.write("|---------------------|-------------------|-------------|---------|-------------|\n")

                for pair_key, data in sorted_anova[:20]:
                    sig = "Yes" if data['significant'] else "No"
                    f.write(f"| {data['categorical_feature']} | {data['numerical_feature']} | {data['f_statistic']:.4f} | {data['p_value']:.4f} | {sig} |\n")

                f.write("\n")

            # ExamResultPercent correlations section
            if 'exam_result_correlations' in self.correlation_results:
                f.write("## ExamResultPercent Correlations (Top Features)\n\n")
                exam_data = self.correlation_results['exam_result_correlations']

                f.write("| Feature | Pearson r | P-Value |\n")
                f.write("|---------|-----------|----------|\n")

                for feat, data in sorted(exam_data.items(), key=lambda x: abs(x[1]['correlation']), reverse=True):
                    f.write(f"| {feat} | {data['correlation']:.4f} | {data['p_value']:.4f} |\n")

                f.write("\n")

            # High inter-feature correlations section
            if 'high_correlations' in self.correlation_results:
                high_corr = self.correlation_results['high_correlations']
                if high_corr:
                    f.write("## High Inter-Feature Correlations (>0.5, Continuous Only)\n\n")
                    f.write("| Feature 1 | Feature 2 | Pearson r |\n")
                    f.write("|-----------|-----------|----------|\n")

                    for pair in high_corr:
                        f.write(f"| {pair['feature1']} | {pair['feature2']} | {pair['correlation']:.4f} |\n")
                    f.write("\n")
                else:
                    f.write("## Inter-Feature Correlations\n\n")
                    f.write("No high inter-feature correlations (>0.5) detected.\n\n")

            # Files generated section
            f.write("## Files Generated\n\n")
            f.write("- `correlation_results.json` - Complete correlation data\n")
            f.write("- `correlation_matrix.csv` - Pearson correlation matrix (continuous + binary variables)\n")
            f.write("- `correlation_matrix.png` - Heatmap visualization\n")
            f.write("- `target_correlations.png` - Point-biserial correlations plot\n")
            f.write("- `categorical_associations.png` - Cramer's V associations plot\n")
            f.write("- `anova_results.png` - One-Way ANOVA results plot\n")
            f.write("- `exam_result_correlations.png` - ExamResultPercent scatter plots\n")
            f.write("- `correlation_report.md` - This report\n")

        print(f"[SAVED] Correlation report: {report_path}")


def main():
    print("="*70)
    print("CORRELATION ANALYSIS FOR SOCIAL WORK EXAM DATASET")
    print("="*70)

    analyzer = CorrelationAnalyzer()

    # Try multiple possible file paths
    csv_files_to_try = [
        'data/sdv_dataset_2.csv',
        '../data/sdv_dataset_2.csv',
        os.path.join('..', 'sdv_dataset_2.csv'),
        'models/data/sdv_dataset_2.csv'
    ]

    df = None
    for csv_file in csv_files_to_try:
        if os.path.exists(csv_file):
            print(f"[FILE] Found CSV at: {csv_file}")
            df = analyzer.load_data(csv_file)
            break

    if df is None:
        print(f"[ERROR] CSV file not found. Tried: {csv_files_to_try}")
        return

    # Run all correlation analyses
    analyzer.compute_correlation_matrix(df)
    analyzer.pearson_correlation_analysis(df)
    analyzer.point_biserial_correlation_analysis(df)
    analyzer.chi_square_and_cramers_v_analysis(df)
    analyzer.cramers_v_categorical_analysis(df)
    analyzer.one_way_anova_analysis(df)

    # Generate all visualizations
    analyzer.visualize_correlation_matrix()
    analyzer.visualize_target_correlations()
    analyzer.visualize_categorical_associations()
    analyzer.visualize_anova_results()
    analyzer.visualize_exam_result_correlations(df)

    # Save results and report
    analyzer.save_results()
    analyzer.generate_report()

    print("\n[COMPLETE] Correlation analysis completed successfully!")
    print("[OUTPUT] Results saved to: correlation_analysis/")


if __name__ == "__main__":
    main()