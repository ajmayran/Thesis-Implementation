import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, spearmanr, pointbiserialr, chi2_contingency, f_oneway, kruskal
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    def __init__(self):
        # Categorical columns (non-numeric, multi-class)
        self.categorical_columns = ['Gender', 'IncomeLevel', 'EmploymentStatus']
        # Numerical columns (predictors only, excluding target)
        self.numerical_columns = [
            'Age', 'StudyHours', 'SleepHours', 'Confidence',
            'MockExamScore', 'GPA', 'InternshipGrade', 'TestAnxiety',
            'SocialSupport', 'MotivationScore', 'EnglishProficiency'
        ]
        # Binary columns (0/1 values)
        self.binary_columns = ['ReviewCenter', 'Scholarship']
        # Target column for regression (continuous)
        self.target_column = 'ExamResultPercent'
        # Secondary binary column for additional analysis
        self.binary_target_column = 'Passed'
        self.correlation_matrix = None
        self.correlation_results = {}

    def load_data(self, file_path):
        """Load CSV data and print basic info."""
        try:
            df = pd.read_csv(file_path)
            print(f"[SUCCESS] Data loaded. Shape: {df.shape}")
            print(f"[INFO] Columns found: {list(df.columns)}")
            print(f"[INFO] Target column: {self.target_column}")
            print(f"[INFO] Target stats: Mean={df[self.target_column].mean():.2f}, "
                  f"Std={df[self.target_column].std():.2f}, "
                  f"Min={df[self.target_column].min():.2f}, "
                  f"Max={df[self.target_column].max():.2f}")
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
        """Compute Pearson correlation matrix for all numerical, binary, and target features."""
        print("\n" + "=" * 70)
        print("CORRELATION MATRIX ANALYSIS (ALL NUMERICAL + BINARY + TARGET)")
        print("=" * 70)

        # All features for the matrix: numerical predictors + binary + target
        all_features = self.numerical_columns + self.binary_columns
        available_features = [col for col in all_features if col in df.columns]

        # Add target column
        if self.target_column in df.columns:
            available_features.append(self.target_column)

        # Add binary target for reference
        if self.binary_target_column in df.columns:
            available_features.append(self.binary_target_column)

        # Remove duplicates while preserving order
        available_features = list(dict.fromkeys(available_features))

        print(f"[INFO] Features in correlation matrix: {available_features}")

        correlation_matrix = df[available_features].corr()
        self.correlation_matrix = correlation_matrix

        # Show correlations with ExamResultPercent (target)
        if self.target_column in correlation_matrix.columns:
            target_corr = correlation_matrix[self.target_column].drop(
                [self.target_column] + ([self.binary_target_column] if self.binary_target_column in correlation_matrix.columns else []),
                errors='ignore'
            ).sort_values(key=abs, ascending=False)
            print(f"\n[TARGET CORRELATION] Features correlated with {self.target_column}:")
            for feature, corr in target_corr.items():
                print(f"   {feature}: r = {corr:.4f}")

        # Detect high inter-feature correlations (excluding target columns)
        exclude_cols = [self.target_column, self.binary_target_column]
        print(f"\n[INTER-FEATURE] High inter-feature correlations (|r| > 0.5):")
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                col_i = correlation_matrix.columns[i]
                col_j = correlation_matrix.columns[j]
                if corr_val > 0.5 and col_i not in exclude_cols and col_j not in exclude_cols:
                    high_corr_pairs.append((col_i, col_j, float(correlation_matrix.iloc[i, j])))

        if high_corr_pairs:
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"   {feat1} <-> {feat2}: r = {corr:.4f}")
        else:
            print("   [OK] No high inter-feature correlations found (no multicollinearity concern)")

        # Store results
        self.correlation_results['correlation_matrix'] = correlation_matrix.to_dict()
        self.correlation_results['high_correlations'] = [
            {'feature1': feat1, 'feature2': feat2, 'correlation': float(corr)}
            for feat1, feat2, corr in high_corr_pairs
        ]

        return correlation_matrix

    def pearson_correlation_with_target(self, df):
        """Compute Pearson correlation between each numerical/binary feature and ExamResultPercent."""
        print("\n" + "=" * 70)
        print("PEARSON CORRELATION (FEATURES -> ExamResultPercent)")
        print("=" * 70)

        if self.target_column not in df.columns:
            print(f"[ERROR] Target column '{self.target_column}' not found")
            return {}

        all_features = self.numerical_columns + self.binary_columns
        available_features = [col for col in all_features if col in df.columns]

        pearson_results = {}

        for col in available_features:
            try:
                valid_data = df[[col, self.target_column]].dropna()
                if len(valid_data) > 2:
                    corr_coef, p_value = pearsonr(valid_data[col], valid_data[self.target_column])
                    pearson_results[col] = {
                        'correlation': float(corr_coef),
                        'p_value': float(p_value),
                        'significant': bool(p_value < 0.05),
                        'n': len(valid_data),
                        'strength': self._interpret_correlation_strength(abs(corr_coef))
                    }
                    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"   {col}: r = {corr_coef:.4f}, p = {p_value:.6f}, n = {len(valid_data)} {sig_marker}")
            except Exception as e:
                print(f"   {col}: Failed ({e})")

        # Sort by absolute correlation
        pearson_results = dict(sorted(pearson_results.items(),
                                       key=lambda x: abs(x[1]['correlation']),
                                       reverse=True))

        self.correlation_results['pearson_with_target'] = pearson_results
        print(f"\n[SUCCESS] Computed Pearson correlations for {len(pearson_results)} features")

        return pearson_results

    def spearman_correlation_with_target(self, df):
        """Compute Spearman rank correlation between each feature and ExamResultPercent.
        Spearman captures non-linear monotonic relationships that Pearson might miss."""
        print("\n" + "=" * 70)
        print("SPEARMAN RANK CORRELATION (FEATURES -> ExamResultPercent)")
        print("=" * 70)

        if self.target_column not in df.columns:
            print(f"[ERROR] Target column '{self.target_column}' not found")
            return {}

        all_features = self.numerical_columns + self.binary_columns
        available_features = [col for col in all_features if col in df.columns]

        spearman_results = {}

        for col in available_features:
            try:
                valid_data = df[[col, self.target_column]].dropna()
                if len(valid_data) > 2:
                    corr_coef, p_value = spearmanr(valid_data[col], valid_data[self.target_column])
                    spearman_results[col] = {
                        'correlation': float(corr_coef),
                        'p_value': float(p_value),
                        'significant': bool(p_value < 0.05),
                        'n': len(valid_data),
                        'strength': self._interpret_correlation_strength(abs(corr_coef))
                    }
                    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"   {col}: rho = {corr_coef:.4f}, p = {p_value:.6f}, n = {len(valid_data)} {sig_marker}")
            except Exception as e:
                print(f"   {col}: Failed ({e})")

        # Sort by absolute correlation
        spearman_results = dict(sorted(spearman_results.items(),
                                        key=lambda x: abs(x[1]['correlation']),
                                        reverse=True))

        self.correlation_results['spearman_with_target'] = spearman_results
        print(f"\n[SUCCESS] Computed Spearman correlations for {len(spearman_results)} features")

        return spearman_results

    def pearson_vs_spearman_comparison(self, df):
        """Compare Pearson and Spearman to detect potential non-linear relationships."""
        print("\n" + "=" * 70)
        print("PEARSON vs SPEARMAN COMPARISON (DETECTING NON-LINEARITY)")
        print("=" * 70)

        pearson_data = self.correlation_results.get('pearson_with_target', {})
        spearman_data = self.correlation_results.get('spearman_with_target', {})

        if not pearson_data or not spearman_data:
            print("[ERROR] Run pearson_correlation_with_target and spearman_correlation_with_target first")
            return {}

        comparison = {}
        print(f"\n   {'Feature':<22} {'Pearson r':>10} {'Spearman rho':>13} {'Difference':>11} {'Non-Linear?':>12}")
        print("   " + "-" * 70)

        for feature in pearson_data:
            if feature in spearman_data:
                p_corr = pearson_data[feature]['correlation']
                s_corr = spearman_data[feature]['correlation']
                diff = abs(abs(s_corr) - abs(p_corr))
                # If Spearman is notably higher than Pearson, suggests non-linear monotonic relationship
                non_linear = diff > 0.05

                comparison[feature] = {
                    'pearson_r': float(p_corr),
                    'spearman_rho': float(s_corr),
                    'absolute_difference': float(diff),
                    'non_linear_indicator': bool(non_linear),
                    'pearson_significant': pearson_data[feature]['significant'],
                    'spearman_significant': spearman_data[feature]['significant']
                }

                flag = " <-- NON-LINEAR" if non_linear else ""
                print(f"   {feature:<22} {p_corr:>10.4f} {s_corr:>13.4f} {diff:>11.4f} {flag}")

        self.correlation_results['pearson_vs_spearman'] = comparison
        print(f"\n[INFO] Features with potential non-linear relationships: "
              f"{sum(1 for v in comparison.values() if v['non_linear_indicator'])}")

        return comparison

    def pairwise_pearson_analysis(self, df):
        """Compute pairwise Pearson correlation for all numerical and binary feature pairs."""
        print("\n" + "=" * 70)
        print("PAIRWISE PEARSON CORRELATION (INTER-FEATURE ANALYSIS)")
        print("=" * 70)

        all_features = self.numerical_columns + self.binary_columns
        available_features = [col for col in all_features if col in df.columns]

        pearson_correlations = {}

        for i, col in enumerate(available_features):
            for other_col in available_features[i + 1:]:
                try:
                    valid_data = df[[col, other_col]].dropna()
                    if len(valid_data) > 2:
                        corr_coef, p_value = pearsonr(valid_data[col], valid_data[other_col])
                        pair_key = f"{col}_vs_{other_col}"
                        pearson_correlations[pair_key] = {
                            'feature1': col,
                            'feature2': other_col,
                            'correlation': float(corr_coef),
                            'p_value': float(p_value),
                            'significant': bool(p_value < 0.05)
                        }
                except Exception as e:
                    print(f"   {col} vs {other_col}: Failed ({e})")

        self.correlation_results['pairwise_pearson'] = pearson_correlations
        print(f"[SUCCESS] Computed {len(pearson_correlations)} pairwise Pearson correlations")

        # Show top 10 strongest
        sorted_pairs = sorted(pearson_correlations.items(),
                             key=lambda x: abs(x[1]['correlation']),
                             reverse=True)[:10]
        print("\n   Top 10 strongest inter-feature correlations:")
        for pair_key, data in sorted_pairs:
            sig = "*" if data['significant'] else ""
            print(f"   {data['feature1']} <-> {data['feature2']}: r = {data['correlation']:.4f} {sig}")

        return pearson_correlations

    def point_biserial_correlation_analysis(self, df):
        """Compute point-biserial correlation between continuous features and binary Passed column.
        This is kept as supplementary analysis since Passed is derived from ExamResultPercent."""
        print("\n" + "=" * 70)
        print("POINT-BISERIAL CORRELATION (FEATURES -> Passed) [SUPPLEMENTARY]")
        print("=" * 70)

        if self.binary_target_column not in df.columns:
            print(f"[ERROR] Binary target column '{self.binary_target_column}' not found")
            return {}

        all_features = self.numerical_columns + self.binary_columns
        available_features = [col for col in all_features if col in df.columns]

        point_biserial_correlations = {}

        for col in available_features:
            try:
                valid_data = df[[col, self.binary_target_column]].dropna()
                if len(valid_data) > 2:
                    corr_coef, p_value = pointbiserialr(valid_data[self.binary_target_column], valid_data[col])
                    point_biserial_correlations[col] = {
                        'correlation': float(corr_coef),
                        'p_value': float(p_value),
                        'significant': bool(p_value < 0.05)
                    }
                    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"   {col}: rpb = {corr_coef:.4f}, p = {p_value:.4f} {sig_marker}")
            except Exception as e:
                print(f"   {col}: Failed ({e})")

        # Sort by absolute correlation
        point_biserial_correlations = dict(sorted(point_biserial_correlations.items(),
                                                   key=lambda x: abs(x[1]['correlation']),
                                                   reverse=True))

        self.correlation_results['point_biserial_correlation'] = point_biserial_correlations

        return point_biserial_correlations

    def chi_square_and_cramers_v_analysis(self, df):
        """Compute Chi-Square and Cramer's V between categorical features and binary Passed."""
        print("\n" + "=" * 70)
        print("CHI-SQUARE TEST + CRAMER'S V (CATEGORICAL -> Passed)")
        print("=" * 70)

        if self.binary_target_column not in df.columns:
            print(f"[ERROR] Binary target column '{self.binary_target_column}' not found")
            return {}

        chi_square_results = {}

        for col in self.categorical_columns:
            if col in df.columns:
                try:
                    contingency_table = pd.crosstab(df[col], df[self.binary_target_column])
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
                    print(f"   {col}: chi2 = {chi2:.4f}, p = {p_value:.4f}, Cramer's V = {cramers_v:.4f} {sig_marker}")
                except Exception as e:
                    print(f"   {col}: Failed ({e})")

        self.correlation_results['chi_square_cramers_v'] = chi_square_results

        return chi_square_results

    def cramers_v_categorical_analysis(self, df):
        """Compute Cramer's V between all pairs of categorical features."""
        print("\n" + "=" * 70)
        print("CRAMER'S V (CATEGORICAL -> CATEGORICAL)")
        print("=" * 70)

        cramers_v_results = {}

        for i, col1 in enumerate(self.categorical_columns):
            if col1 in df.columns:
                for col2 in self.categorical_columns[i + 1:]:
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
                            print(f"   {col1} vs {col2}: Cramer's V = {cramers_v:.4f}, p = {p_value:.4f} {sig_marker}")
                        except Exception as e:
                            print(f"   {col1} vs {col2}: Failed ({e})")

        self.correlation_results['cramers_v_categorical'] = cramers_v_results

        return cramers_v_results

    def one_way_anova_analysis(self, df):
        """Compute One-Way ANOVA between categorical features and ExamResultPercent (and other numerical)."""
        print("\n" + "=" * 70)
        print("ONE-WAY ANOVA (CATEGORICAL -> CONTINUOUS)")
        print("=" * 70)

        # Include target in numerical columns for ANOVA
        num_cols_with_target = self.numerical_columns + [self.target_column]

        anova_results = {}

        for cat_col in self.categorical_columns:
            if cat_col not in df.columns:
                continue

            for num_col in num_cols_with_target:
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

                        # Also compute eta-squared (effect size)
                        all_data = df[[cat_col, num_col]].dropna()
                        grand_mean = all_data[num_col].mean()
                        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
                        ss_total = sum((all_data[num_col] - grand_mean) ** 2)
                        eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0

                        pair_key = f"{cat_col}_vs_{num_col}"
                        anova_results[pair_key] = {
                            'categorical_feature': cat_col,
                            'numerical_feature': num_col,
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'eta_squared': eta_squared,
                            'significant': bool(p_value < 0.05),
                            'n_groups': len(groups),
                            'effect_size': self._interpret_eta_squared(eta_squared)
                        }

                        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                        print(f"   {cat_col} -> {num_col}: F = {f_stat:.4f}, p = {p_value:.4f}, "
                              f"eta2 = {eta_squared:.4f} {sig_marker}")

                except Exception as e:
                    print(f"   {cat_col} -> {num_col}: Failed ({e})")

        self.correlation_results['one_way_anova'] = anova_results
        print(f"\n[SUCCESS] Completed {len(anova_results)} ANOVA tests")

        # Highlight ANOVA results specifically for ExamResultPercent
        print(f"\n   ANOVA results for {self.target_column}:")
        for key, data in anova_results.items():
            if data['numerical_feature'] == self.target_column:
                sig = "SIGNIFICANT" if data['significant'] else "NOT significant"
                print(f"   {data['categorical_feature']} -> {self.target_column}: "
                      f"F = {data['f_statistic']:.4f}, p = {data['p_value']:.4f}, "
                      f"eta2 = {data['eta_squared']:.4f} ({sig})")

        return anova_results

    def kruskal_wallis_analysis(self, df):
        """Compute Kruskal-Wallis H test (non-parametric alternative to ANOVA)
        between categorical features and ExamResultPercent."""
        print("\n" + "=" * 70)
        print("KRUSKAL-WALLIS H TEST (CATEGORICAL -> ExamResultPercent) [NON-PARAMETRIC]")
        print("=" * 70)

        if self.target_column not in df.columns:
            print(f"[ERROR] Target column '{self.target_column}' not found")
            return {}

        kruskal_results = {}

        for cat_col in self.categorical_columns:
            if cat_col not in df.columns:
                continue

            try:
                groups = []
                categories = df[cat_col].dropna().unique()

                for category in categories:
                    group_data = df[df[cat_col] == category][self.target_column].dropna().values
                    if len(group_data) > 0:
                        groups.append(group_data)

                if len(groups) >= 2:
                    h_stat, p_value = kruskal(*groups)

                    kruskal_results[cat_col] = {
                        'h_statistic': float(h_stat),
                        'p_value': float(p_value),
                        'significant': bool(p_value < 0.05),
                        'n_groups': len(groups)
                    }

                    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"   {cat_col}: H = {h_stat:.4f}, p = {p_value:.4f} {sig_marker}")

            except Exception as e:
                print(f"   {cat_col}: Failed ({e})")

        self.correlation_results['kruskal_wallis'] = kruskal_results

        return kruskal_results

    # -- Visualization Methods --

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
        plt.title('Feature Correlation Matrix (Pearson r)\nAll Numerical, Binary, and Target Variables',
                  fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Correlation matrix heatmap: {output_path}")
        plt.close()

    def visualize_pearson_spearman_comparison(self, output_dir='correlation_analysis'):
        """Generate side-by-side Pearson vs Spearman bar chart."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pearson_data = self.correlation_results.get('pearson_with_target', {})
        spearman_data = self.correlation_results.get('spearman_with_target', {})

        if not pearson_data or not spearman_data:
            print("[ERROR] Run pearson and spearman analyses first")
            return

        # Get common features sorted by absolute Pearson correlation
        common_features = [f for f in pearson_data if f in spearman_data]
        common_features.sort(key=lambda f: abs(pearson_data[f]['correlation']), reverse=True)

        pearson_vals = [pearson_data[f]['correlation'] for f in common_features]
        spearman_vals = [spearman_data[f]['correlation'] for f in common_features]
        pearson_sig = [pearson_data[f]['significant'] for f in common_features]
        spearman_sig = [spearman_data[f]['significant'] for f in common_features]

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # Panel 1: Pearson correlation
        colors_p = ['#2ecc71' if sig else '#95a5a6' for sig in pearson_sig]
        axes[0].barh(common_features, pearson_vals, color=colors_p)
        axes[0].set_xlabel('Pearson r', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Features', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Pearson Correlation with {self.target_column}', fontsize=13, fontweight='bold')
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[0].grid(axis='x', alpha=0.3)
        axes[0].invert_yaxis()

        # Panel 2: Spearman correlation
        colors_s = ['#3498db' if sig else '#95a5a6' for sig in spearman_sig]
        axes[1].barh(common_features, spearman_vals, color=colors_s)
        axes[1].set_xlabel('Spearman rho', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Spearman Correlation with {self.target_column}', fontsize=13, fontweight='bold')
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].invert_yaxis()

        # Panel 3: Absolute difference (non-linearity indicator)
        diffs = [abs(abs(s) - abs(p)) for p, s in zip(pearson_vals, spearman_vals)]
        diff_colors = ['#e74c3c' if d > 0.05 else '#95a5a6' for d in diffs]
        axes[2].barh(common_features, diffs, color=diff_colors)
        axes[2].set_xlabel('|Spearman| - |Pearson|', fontsize=12, fontweight='bold')
        axes[2].set_title('Non-Linearity Indicator\n(Difference > 0.05 suggests non-linear)', fontsize=13, fontweight='bold')
        axes[2].axvline(x=0.05, color='red', linestyle='--', linewidth=1.5, label='Threshold (0.05)')
        axes[2].grid(axis='x', alpha=0.3)
        axes[2].legend()
        axes[2].invert_yaxis()

        plt.suptitle('Pearson vs Spearman Correlation Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'pearson_vs_spearman.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Pearson vs Spearman comparison: {output_path}")
        plt.close()

    def visualize_target_correlations(self, output_dir='correlation_analysis'):
        """Generate combined bar chart showing Pearson and Spearman correlations with target."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pearson_data = self.correlation_results.get('pearson_with_target', {})
        if not pearson_data:
            print("[ERROR] No Pearson correlation results with target")
            return

        features = list(pearson_data.keys())
        correlations = [pearson_data[feat]['correlation'] for feat in features]
        p_values = [pearson_data[feat]['p_value'] for feat in features]

        # Sort by absolute correlation
        sorted_indices = np.argsort([abs(c) for c in correlations])[::-1]
        features_sorted = [features[i] for i in sorted_indices]
        correlations_sorted = [correlations[i] for i in sorted_indices]
        p_values_sorted = [p_values[i] for i in sorted_indices]

        colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in p_values_sorted]

        plt.figure(figsize=(12, 9))
        bars = plt.barh(features_sorted, correlations_sorted, color=colors)

        # Add correlation values on bars
        for bar, corr, p_val in zip(bars, correlations_sorted, p_values_sorted):
            x_pos = bar.get_width()
            offset = 0.003 if x_pos >= 0 else -0.003
            ha = 'left' if x_pos >= 0 else 'right'
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            plt.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                    f'{corr:.4f}{sig}', va='center', ha=ha, fontsize=9)

        plt.xlabel('Pearson Correlation Coefficient (r)', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title(f'Feature Correlations with {self.target_column} (Pearson r)',
                  fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3)

        plt.legend([plt.Rectangle((0, 0), 1, 1, color='#2ecc71'),
                    plt.Rectangle((0, 0), 1, 1, color='#95a5a6')],
                   ['Significant (p < 0.05)', 'Not Significant'],
                   loc='lower right')

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'target_correlations_pearson.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Target correlations (Pearson): {output_path}")
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
        plt.title(f"Categorical Features Association with {self.binary_target_column} (Cramer's V)",
                  fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        plt.legend([plt.Rectangle((0, 0), 1, 1, color='#3498db'),
                    plt.Rectangle((0, 0), 1, 1, color='#95a5a6')],
                   ['Significant (p < 0.05)', 'Not Significant'],
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
        eta_sqs = []

        for pair_key, data in anova_data.items():
            pairs.append(f"{data['categorical_feature']} -> {data['numerical_feature']}")
            f_stats.append(data['f_statistic'])
            p_values.append(data['p_value'])
            eta_sqs.append(data.get('eta_squared', 0))

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

        plt.legend([plt.Rectangle((0, 0), 1, 1, color='#e74c3c'),
                    plt.Rectangle((0, 0), 1, 1, color='#95a5a6')],
                   ['Significant (p < 0.05)', 'Not Significant'],
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

        if self.target_column not in df.columns:
            print(f"[ERROR] {self.target_column} column not found")
            return

        pearson_data = self.correlation_results.get('pearson_with_target', {})
        spearman_data = self.correlation_results.get('spearman_with_target', {})

        # Get top 6 features by absolute Pearson correlation
        sorted_features = sorted(pearson_data.items(),
                                 key=lambda x: abs(x[1]['correlation']),
                                 reverse=True)[:6]

        if not sorted_features:
            print("[INFO] No valid correlations found for scatter plots")
            return

        has_passed = self.binary_target_column in df.columns

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (feat, p_data) in enumerate(sorted_features):
            ax = axes[idx]
            valid = df[[self.target_column, feat]].dropna()
            if has_passed:
                valid = valid.join(df[self.binary_target_column])
                colors = valid[self.binary_target_column].map({1: '#2ecc71', 0: '#e74c3c'})
                ax.scatter(valid[feat], valid[self.target_column], c=colors, alpha=0.6,
                          edgecolors='white', linewidth=0.5)
            else:
                ax.scatter(valid[feat], valid[self.target_column], color='steelblue', alpha=0.6,
                          edgecolors='white', linewidth=0.5)

            # Add regression line
            z = np.polyfit(valid[feat], valid[self.target_column], 1)
            p = np.poly1d(z)
            x_sorted = sorted(valid[feat])
            ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2)

            ax.set_xlabel(feat, fontsize=10, fontweight='bold')
            ax.set_ylabel(self.target_column, fontsize=10, fontweight='bold')

            # Show both Pearson and Spearman in title
            s_data = spearman_data.get(feat, {})
            s_corr = s_data.get('correlation', 'N/A')
            title_text = f"r={p_data['correlation']:.3f}, p={p_data['p_value']:.4f}"
            if isinstance(s_corr, float):
                title_text += f"\nrho={s_corr:.3f}"
            ax.set_title(title_text, fontsize=10)
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(len(sorted_features), len(axes)):
            axes[idx].set_visible(False)

        color_label = ' (Green=Passed, Red=Failed)' if has_passed else ''
        fig.suptitle(f'{self.target_column} vs Top Correlated Features{color_label}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'exam_result_correlations.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] ExamResultPercent correlation scatter plots: {output_path}")
        plt.close()

        # Store ExamResultPercent correlations summary in results
        self.correlation_results['exam_result_correlations'] = {
            feat: {'pearson_r': float(data['correlation']),
                   'pearson_p': float(data['p_value']),
                   'spearman_rho': float(spearman_data[feat]['correlation']) if feat in spearman_data else None,
                   'spearman_p': float(spearman_data[feat]['p_value']) if feat in spearman_data else None}
            for feat, data in sorted_features
        }

    def visualize_anova_target_boxplots(self, df, output_dir='correlation_analysis'):
        """Generate box plots showing ExamResultPercent distribution by categorical feature levels."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        available_cats = [col for col in self.categorical_columns if col in df.columns]
        if not available_cats:
            print("[ERROR] No categorical columns found")
            return

        fig, axes = plt.subplots(1, len(available_cats), figsize=(7 * len(available_cats), 6))
        if len(available_cats) == 1:
            axes = [axes]

        for idx, cat_col in enumerate(available_cats):
            ax = axes[idx]
            categories = sorted(df[cat_col].dropna().unique())
            data_groups = [df[df[cat_col] == cat][self.target_column].dropna() for cat in categories]

            bp = ax.boxplot(data_groups, labels=categories, patch_artist=True, showmeans=True)

            colors_box = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
            for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Get ANOVA result for this categorical vs target
            anova_key = f"{cat_col}_vs_{self.target_column}"
            anova_data = self.correlation_results.get('one_way_anova', {}).get(anova_key, {})
            f_stat = anova_data.get('f_statistic', 'N/A')
            p_val = anova_data.get('p_value', 'N/A')
            eta2 = anova_data.get('eta_squared', 'N/A')

            title_text = f"{cat_col}"
            if isinstance(f_stat, float):
                title_text += f"\nF={f_stat:.3f}, p={p_val:.4f}, eta2={eta2:.4f}"
            ax.set_title(title_text, fontsize=11, fontweight='bold')
            ax.set_ylabel(self.target_column, fontsize=10)
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle(f'{self.target_column} Distribution by Categorical Features (ANOVA)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'anova_target_boxplots.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] ANOVA target box plots: {output_path}")
        plt.close()

    # -- Helper Methods --

    def _interpret_correlation_strength(self, abs_corr):
        """Interpret correlation coefficient strength based on Cohen's guidelines."""
        if abs_corr >= 0.5:
            return "Strong"
        elif abs_corr >= 0.3:
            return "Moderate"
        elif abs_corr >= 0.1:
            return "Weak"
        else:
            return "Negligible"

    def _interpret_eta_squared(self, eta_sq):
        """Interpret eta-squared effect size based on Cohen's guidelines."""
        if eta_sq >= 0.14:
            return "Large"
        elif eta_sq >= 0.06:
            return "Medium"
        elif eta_sq >= 0.01:
            return "Small"
        else:
            return "Negligible"

    # -- Save and Report Methods --

    def save_results(self, output_dir='correlation_analysis'):
        """Save correlation results to JSON and CSV files."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save JSON (handle non-serializable types)
        json_path = os.path.join(output_dir, 'correlation_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.correlation_results, f, indent=2, default=str)
        print(f"[SAVED] Correlation results: {json_path}")

        # Save correlation matrix CSV
        if self.correlation_matrix is not None:
            csv_path = os.path.join(output_dir, 'correlation_matrix.csv')
            self.correlation_matrix.to_csv(csv_path)
            print(f"[SAVED] Correlation matrix CSV: {csv_path}")

    def generate_report(self, output_dir='correlation_analysis'):
        """Generate a thesis-ready markdown report summarizing all correlation analysis results."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        report_path = os.path.join(output_dir, 'correlation_report.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Correlation Analysis Report\n\n")
            f.write("## Overview\n\n")
            f.write(f"**Target Variable**: `{self.target_column}` (Continuous - Exam Score Percentage)\n\n")
            f.write(f"**Secondary Variable**: `{self.binary_target_column}` (Binary - Pass/Fail, threshold: 70%)\n\n")
            f.write("### Feature Groups\n\n")
            f.write(f"- **Numerical Features** ({len(self.numerical_columns)}): {', '.join(self.numerical_columns)}\n")
            f.write(f"- **Binary Features** ({len(self.binary_columns)}): {', '.join(self.binary_columns)}\n")
            f.write(f"- **Categorical Features** ({len(self.categorical_columns)}): {', '.join(self.categorical_columns)}\n\n")

            f.write("## Statistical Methods Applied\n\n")
            f.write("| Method | Variable Types | Purpose |\n")
            f.write("|--------|---------------|----------|\n")
            f.write("| Pearson Correlation (r) | Continuous/Binary -> Continuous Target | Measures linear relationship strength |\n")
            f.write("| Spearman Rank Correlation (rho) | Continuous/Binary -> Continuous Target | Captures monotonic (including non-linear) relationships |\n")
            f.write("| Point-Biserial Correlation (rpb) | Continuous -> Binary (Passed) | Supplementary analysis for pass/fail outcome |\n")
            f.write("| Chi-Square + Cramer's V | Categorical -> Binary (Passed) | Tests categorical feature association with pass/fail |\n")
            f.write("| One-Way ANOVA + Eta-squared | Categorical -> Continuous Target | Tests mean score differences across categorical groups |\n")
            f.write("| Kruskal-Wallis H | Categorical -> Continuous Target | Non-parametric alternative to ANOVA |\n")
            f.write("| Cramer's V (Inter-feature) | Categorical -> Categorical | Checks categorical feature independence |\n\n")

            # Pearson correlation with target section
            if 'pearson_with_target' in self.correlation_results:
                f.write(f"## Pearson Correlation with {self.target_column}\n\n")
                p_data = self.correlation_results['pearson_with_target']
                f.write("| Rank | Feature | Pearson r | P-Value | Significant | Strength |\n")
                f.write("|------|---------|-----------|---------|-------------|----------|\n")

                for rank, (feature, data) in enumerate(p_data.items(), 1):
                    sig = "Yes" if data['significant'] else "No"
                    f.write(f"| {rank} | {feature} | {data['correlation']:.4f} | "
                            f"{data['p_value']:.6f} | {sig} | {data['strength']} |\n")
                f.write("\n")

            # Spearman correlation with target section
            if 'spearman_with_target' in self.correlation_results:
                f.write(f"## Spearman Rank Correlation with {self.target_column}\n\n")
                s_data = self.correlation_results['spearman_with_target']
                f.write("| Rank | Feature | Spearman rho | P-Value | Significant | Strength |\n")
                f.write("|------|---------|--------------|---------|-------------|----------|\n")

                for rank, (feature, data) in enumerate(s_data.items(), 1):
                    sig = "Yes" if data['significant'] else "No"
                    f.write(f"| {rank} | {feature} | {data['correlation']:.4f} | "
                            f"{data['p_value']:.6f} | {sig} | {data['strength']} |\n")
                f.write("\n")

            # Pearson vs Spearman comparison
            if 'pearson_vs_spearman' in self.correlation_results:
                f.write("## Pearson vs Spearman Comparison (Non-Linearity Detection)\n\n")
                comp_data = self.correlation_results['pearson_vs_spearman']
                f.write("| Feature | Pearson r | Spearman rho | Abs Difference | Non-Linear? |\n")
                f.write("|---------|-----------|--------------|----------------|-------------|\n")

                for feature, data in sorted(comp_data.items(),
                                           key=lambda x: x[1]['absolute_difference'],
                                           reverse=True):
                    nl = "Yes" if data['non_linear_indicator'] else "No"
                    f.write(f"| {feature} | {data['pearson_r']:.4f} | {data['spearman_rho']:.4f} | "
                            f"{data['absolute_difference']:.4f} | {nl} |\n")
                f.write("\n")

            # Point-Biserial section (supplementary)
            if 'point_biserial_correlation' in self.correlation_results:
                f.write(f"## Point-Biserial Correlation (Features -> {self.binary_target_column}) [Supplementary]\n\n")
                pb_data = self.correlation_results['point_biserial_correlation']
                f.write("| Feature | Correlation (rpb) | P-Value | Significant |\n")
                f.write("|---------|-------------------|---------|-------------|\n")

                for feature, data in pb_data.items():
                    sig = "Yes" if data['significant'] else "No"
                    f.write(f"| {feature} | {data['correlation']:.4f} | {data['p_value']:.4f} | {sig} |\n")
                f.write("\n")

            # Chi-Square section
            if 'chi_square_cramers_v' in self.correlation_results:
                f.write(f"## Chi-Square Test + Cramer's V (Categorical -> {self.binary_target_column})\n\n")
                chi_data = self.correlation_results['chi_square_cramers_v']
                sorted_features = sorted(chi_data.items(),
                                        key=lambda x: x[1]['cramers_v'],
                                        reverse=True)

                f.write("| Feature | Chi-Square | P-Value | Cramer's V | Significant |\n")
                f.write("|---------|------------|---------|------------|-------------|\n")

                for feature, data in sorted_features:
                    sig = "Yes" if data['significant'] else "No"
                    f.write(f"| {feature} | {data['chi2']:.4f} | {data['p_value']:.4f} | "
                            f"{data['cramers_v']:.4f} | {sig} |\n")
                f.write("\n")

            # ANOVA section
            if 'one_way_anova' in self.correlation_results:
                f.write(f"## One-Way ANOVA (Categorical -> Continuous)\n\n")

                # First show ANOVA for target
                f.write(f"### ANOVA for {self.target_column}\n\n")
                anova_data = self.correlation_results['one_way_anova']

                f.write("| Categorical Feature | F-Statistic | P-Value | Eta-Squared | Effect Size | Significant |\n")
                f.write("|---------------------|-------------|---------|-------------|-------------|-------------|\n")

                for pair_key, data in sorted(anova_data.items(),
                                            key=lambda x: x[1]['f_statistic'],
                                            reverse=True):
                    if data['numerical_feature'] == self.target_column:
                        sig = "Yes" if data['significant'] else "No"
                        f.write(f"| {data['categorical_feature']} | {data['f_statistic']:.4f} | "
                                f"{data['p_value']:.4f} | {data.get('eta_squared', 0):.4f} | "
                                f"{data.get('effect_size', 'N/A')} | {sig} |\n")
                f.write("\n")

                # Then show all ANOVA results
                f.write("### All ANOVA Results (Top 20 by F-Statistic)\n\n")
                f.write("| Categorical Feature | Numerical Feature | F-Statistic | P-Value | Eta-Squared | Significant |\n")
                f.write("|---------------------|-------------------|-------------|---------|-------------|-------------|\n")

                sorted_anova = sorted(anova_data.items(),
                                     key=lambda x: x[1]['f_statistic'],
                                     reverse=True)

                for pair_key, data in sorted_anova[:20]:
                    sig = "Yes" if data['significant'] else "No"
                    f.write(f"| {data['categorical_feature']} | {data['numerical_feature']} | "
                            f"{data['f_statistic']:.4f} | {data['p_value']:.4f} | "
                            f"{data.get('eta_squared', 0):.4f} | {sig} |\n")
                f.write("\n")

            # Kruskal-Wallis section
            if 'kruskal_wallis' in self.correlation_results:
                kw_data = self.correlation_results['kruskal_wallis']
                if kw_data:
                    f.write(f"## Kruskal-Wallis H Test (Categorical -> {self.target_column}) [Non-Parametric]\n\n")
                    f.write("| Feature | H-Statistic | P-Value | Significant |\n")
                    f.write("|---------|-------------|---------|-------------|\n")

                    for feature, data in sorted(kw_data.items(),
                                               key=lambda x: x[1]['h_statistic'],
                                               reverse=True):
                        sig = "Yes" if data['significant'] else "No"
                        f.write(f"| {feature} | {data['h_statistic']:.4f} | "
                                f"{data['p_value']:.4f} | {sig} |\n")
                    f.write("\n")

            # Cramer's V categorical section
            if 'cramers_v_categorical' in self.correlation_results:
                cramers_data = self.correlation_results['cramers_v_categorical']
                if cramers_data:
                    f.write("## Cramer's V (Categorical Inter-Feature Analysis)\n\n")
                    f.write("| Feature 1 | Feature 2 | Cramer's V | P-Value | Significant |\n")
                    f.write("|-----------|-----------|------------|---------|-------------|\n")

                    for pair_key, data in sorted(cramers_data.items(),
                                                key=lambda x: x[1]['cramers_v'],
                                                reverse=True):
                        sig = "Yes" if data['significant'] else "No"
                        f.write(f"| {data['feature1']} | {data['feature2']} | "
                                f"{data['cramers_v']:.4f} | {data['p_value']:.4f} | {sig} |\n")
                    f.write("\n")

            # Inter-feature correlations section
            if 'high_correlations' in self.correlation_results:
                high_corr = self.correlation_results['high_correlations']
                f.write("## Inter-Feature Correlations (Multicollinearity Check)\n\n")
                if high_corr:
                    f.write("**WARNING**: High inter-feature correlations detected (|r| > 0.5):\n\n")
                    f.write("| Feature 1 | Feature 2 | Pearson r |\n")
                    f.write("|-----------|-----------|----------|\n")
                    for pair in high_corr:
                        f.write(f"| {pair['feature1']} | {pair['feature2']} | {pair['correlation']:.4f} |\n")
                else:
                    f.write("No high inter-feature correlations (|r| > 0.5) detected among predictor variables. "
                            "This confirms that multicollinearity is not a concern in the dataset.\n")
                f.write("\n")

            # ExamResultPercent top correlations
            if 'exam_result_correlations' in self.correlation_results:
                f.write(f"## {self.target_column} Top Correlated Features (Summary)\n\n")
                exam_data = self.correlation_results['exam_result_correlations']

                f.write("| Feature | Pearson r | Pearson p | Spearman rho | Spearman p |\n")
                f.write("|---------|-----------|-----------|--------------|------------|\n")

                for feat, data in sorted(exam_data.items(),
                                        key=lambda x: abs(x[1]['pearson_r']),
                                        reverse=True):
                    s_rho = f"{data['spearman_rho']:.4f}" if data['spearman_rho'] is not None else "N/A"
                    s_p = f"{data['spearman_p']:.6f}" if data['spearman_p'] is not None else "N/A"
                    f.write(f"| {feat} | {data['pearson_r']:.4f} | {data['pearson_p']:.6f} | "
                            f"{s_rho} | {s_p} |\n")
                f.write("\n")

            # Files generated section
            f.write("## Files Generated\n\n")
            f.write("| File | Description |\n")
            f.write("|------|-------------|\n")
            f.write("| `correlation_results.json` | Complete correlation data (all methods) |\n")
            f.write("| `correlation_matrix.csv` | Pearson correlation matrix |\n")
            f.write("| `correlation_matrix.png` | Correlation matrix heatmap |\n")
            f.write("| `target_correlations_pearson.png` | Pearson correlations with target bar chart |\n")
            f.write("| `pearson_vs_spearman.png` | Pearson vs Spearman comparison (3-panel) |\n")
            f.write("| `categorical_associations.png` | Cramer's V associations bar chart |\n")
            f.write("| `anova_results.png` | One-Way ANOVA results bar chart |\n")
            f.write("| `anova_target_boxplots.png` | Box plots of target by categorical features |\n")
            f.write("| `exam_result_correlations.png` | Scatter plots of target vs top features |\n")
            f.write("| `correlation_report.md` | This report |\n")

        print(f"[SAVED] Correlation report: {report_path}")


def main():
    print("=" * 70)
    print("CORRELATION ANALYSIS FOR SOCIAL WORK EXAM DATASET")
    print("Target: ExamResultPercent (Continuous)")
    print("Methods: Pearson, Spearman, Point-Biserial, Chi-Square, ANOVA, Kruskal-Wallis")
    print("=" * 70)

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

    # -- Run all correlation analyses --

    # 1. Correlation matrix for all numerical + binary + target
    analyzer.compute_correlation_matrix(df)

    # 2. Pearson correlation: each feature vs ExamResultPercent (primary)
    analyzer.pearson_correlation_with_target(df)

    # 3. Spearman correlation: each feature vs ExamResultPercent (captures non-linear)
    analyzer.spearman_correlation_with_target(df)

    # 4. Pearson vs Spearman comparison (detect non-linearity)
    analyzer.pearson_vs_spearman_comparison(df)

    # 5. Pairwise Pearson (inter-feature multicollinearity check)
    analyzer.pairwise_pearson_analysis(df)

    # 6. Point-biserial: features vs Passed (supplementary)
    analyzer.point_biserial_correlation_analysis(df)

    # 7. Chi-Square + Cramer's V: categorical features vs Passed
    analyzer.chi_square_and_cramers_v_analysis(df)

    # 8. Cramer's V: categorical inter-feature independence
    analyzer.cramers_v_categorical_analysis(df)

    # 9. One-Way ANOVA: categorical features vs continuous features (including target)
    analyzer.one_way_anova_analysis(df)

    # 10. Kruskal-Wallis: non-parametric alternative to ANOVA for target
    analyzer.kruskal_wallis_analysis(df)

    # -- Generate all visualizations --
    analyzer.visualize_correlation_matrix()
    analyzer.visualize_target_correlations()
    analyzer.visualize_pearson_spearman_comparison()
    analyzer.visualize_categorical_associations()
    analyzer.visualize_anova_results()
    analyzer.visualize_exam_result_correlations(df)
    analyzer.visualize_anova_target_boxplots(df)

    # -- Save results and report --
    analyzer.save_results()
    analyzer.generate_report()

    print("\n" + "=" * 70)
    print("[COMPLETE] Correlation analysis completed successfully!")
    print(f"[TARGET] Primary target: {analyzer.target_column} (Continuous)")
    print(f"[METHODS] Pearson, Spearman, Point-Biserial, Chi-Square, ANOVA, Kruskal-Wallis")
    print("[OUTPUT] Results saved to: correlation_analysis/")
    print("=" * 70)


if __name__ == "__main__":
    main()