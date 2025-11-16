import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SocialWorkDataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.feature_names = []
        self.categorical_columns = ['Gender', 'IncomeLevel', 'EmploymentStatus']
        self.numerical_columns = ['Age', 'StudyHours', 'SleepHours', 'Confidence', 
                                'MockExamScore', 'GPA', 'InternshipGrade', 'TestAnxiety']
        self.binary_columns = ['ReviewCenter', 'Scholarship']
        self.target_column = 'Passed'
        
        self.data_stats = {}
        self.feature_importance_scores = {} 
        
    def load_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"[SUCCESS] Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"[ERROR] Error loading CSV file: {e}")
            return None
    
    def explore_data(self, df):
        print("\n" + "="*60)
        print("[INFO] DATA EXPLORATION REPORT")
        print("="*60)
        
        print(f"\n[INFO] Dataset Shape: {df.shape}")
        print(f"[INFO] Columns: {list(df.columns)}")
        
        missing_info = df.isnull().sum()
        print(f"\n[CHECK] Missing Values:")
        for col, missing_count in missing_info.items():
            if missing_count > 0:
                print(f"   {col}: {missing_count} ({missing_count/len(df)*100:.2f}%)")
        if missing_info.sum() == 0:
            print("   [SUCCESS] No missing values found!")
        
        print(f"\n[INFO] Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"   {col}: {dtype}")
        
        if self.target_column in df.columns:
            target_dist = df[self.target_column].value_counts()
            target_pct = df[self.target_column].value_counts(normalize=True) * 100
            print(f"\n[TARGET] Target Distribution ({self.target_column}):")
            for val in target_dist.index:
                print(f"   {val}: {target_dist[val]} ({target_pct[val]:.1f}%)")
            print(f"   Pass Rate: {df[self.target_column].mean():.2%}")
        
        print(f"\n[ANALYSIS] Categorical Variables Analysis:")
        for col in self.categorical_columns:
            if col in df.columns:
                unique_vals = df[col].nunique()
                values = df[col].value_counts()
                print(f"   {col}: {unique_vals} unique values")
                for val, count in values.items():
                    print(f"      {val}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"\n[STATS] Numerical Variables Statistics:")
        numerical_stats = df[self.numerical_columns + self.binary_columns].describe()
        print(numerical_stats.round(2))
        
        self.data_stats = {
            'shape': df.shape,
            'missing_values': missing_info.to_dict(),
            'target_distribution': df[self.target_column].value_counts().to_dict() if self.target_column in df.columns else {},
            'pass_rate': df[self.target_column].mean() if self.target_column in df.columns else None,
            'categorical_stats': {},
            'numerical_stats': numerical_stats.to_dict()
        }
        
        for col in self.categorical_columns:
            if col in df.columns:
                self.data_stats['categorical_stats'][col] = df[col].value_counts().to_dict()
        
        return df
    
    def handle_missing_values(self, df):
        print("\n[CLEANING] Handling missing values...")
        
        missing_before = df.isnull().sum().sum()
        
        if missing_before == 0:
            print("   [SUCCESS] No missing values to handle")
            return df
        
        df_clean = df.copy()
        
        if df_clean[self.target_column].isnull().sum() > 0:
            df_clean = df_clean.dropna(subset=[self.target_column])
            print(f"   [CLEAN] Removed rows with missing target")
        
        for col in self.numerical_columns:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"   [FILL] Filled {col} missing values with median: {median_val}")
        
        for col in self.categorical_columns:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"   [FILL] Filled {col} missing values with mode: {mode_val}")
        
        for col in self.binary_columns:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"   [FILL] Filled {col} missing values with mode: {mode_val}")
        
        missing_after = df_clean.isnull().sum().sum()
        print(f"   [SUCCESS] Missing values: {missing_before} -> {missing_after}")
        
        return df_clean
    
    def feature_importance_analysis(self, df):
        print("\n" + "="*60)
        print("[IMPORTANCE] FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        feature_columns = self.categorical_columns + self.numerical_columns + self.binary_columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].copy()
        y = df[self.target_column].values
        
        print("\n[PREPROCESSING] Handling missing values for feature importance analysis...")
        
        for col in self.numerical_columns:
            if col in X.columns and X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
                print(f"   [FILL] Filled {col} missing values with median: {median_val}")
        
        for col in self.categorical_columns:
            if col in X.columns and X[col].isnull().sum() > 0:
                mode_val = X[col].mode()[0]
                X[col].fillna(mode_val, inplace=True)
                print(f"   [FILL] Filled {col} missing values with mode: {mode_val}")
        
        for col in self.binary_columns:
            if col in X.columns and X[col].isnull().sum() > 0:
                mode_val = X[col].mode()[0]
                X[col].fillna(mode_val, inplace=True)
                print(f"   [FILL] Filled {col} missing values with mode: {mode_val}")
        
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"[WARNING] Still have {missing_count} missing values, dropping affected rows...")
            missing_indices = X.isnull().any(axis=1)
            X = X[~missing_indices]
            y = y[~missing_indices]
            print(f"[INFO] Remaining samples after dropping: {len(X)}")
        
        print("\n[ENCODING] Encoding categorical variables...")
        le = LabelEncoder()
        for col in self.categorical_columns:
            if col in X.columns:
                X[col] = le.fit_transform(X[col].astype(str))
                print(f"   [ENCODED] {col}")
        
        if X.isnull().sum().sum() > 0 or np.isnan(X.values).any():
            print("[ERROR] Still have NaN values after preprocessing!")
            print("Missing values by column:")
            print(X.isnull().sum())
            return self.feature_importance_scores
        
        print(f"[SUCCESS] Data ready for feature importance analysis. Shape: {X.shape}")
        
        try:
            print("\n[ANOVA] Running ANOVA F-test...")
            selector_f = SelectKBest(score_func=f_classif, k='all')
            selector_f.fit(X, y)
            f_scores = dict(zip(X.columns, selector_f.scores_))
            self.feature_importance_scores['anova_f_test'] = f_scores
            
            print("\n[ANOVA] ANOVA F-test scores:")
            for feature, score in sorted(f_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"   {feature}: {score:.4f}")
        except Exception as e:
            print(f"[ERROR] ANOVA F-test failed: {e}")
        
        try:
            print("\n[MUTUAL INFO] Running Mutual Information...")
            selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
            selector_mi.fit(X, y)
            mi_scores = dict(zip(X.columns, selector_mi.scores_))
            self.feature_importance_scores['mutual_info'] = mi_scores
            
            print("\n[MUTUAL INFO] Mutual Information scores:")
            for feature, score in sorted(mi_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"   {feature}: {score:.4f}")
        except Exception as e:
            print(f"[ERROR] Mutual Information failed: {e}")
        
        print("\n[CHI-SQUARE] Chi-square tests (categorical vs target):")
        for col in self.categorical_columns:
            if col in df.columns:
                try:
                    contingency_table = pd.crosstab(df[col], df[self.target_column])
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    print(f"   {col}: chi2={chi2:.4f}, p-value={p_value:.4f}")
                    
                    if 'chi_square' not in self.feature_importance_scores:
                        self.feature_importance_scores['chi_square'] = {}
                    self.feature_importance_scores['chi_square'][col] = {
                        'chi2': chi2,
                        'p_value': p_value
                    }
                except Exception as e:
                    print(f"   {col}: Failed ({e})")
        
        return self.feature_importance_scores
    
    def detect_outliers(self, df):
        print("\n[OUTLIERS] Outlier Detection (using IQR method):")
        
        outlier_info = {}
        
        for col in self.numerical_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(df)) * 100
                
                outlier_info[col] = {
                    'count': outlier_count,
                    'percentage': outlier_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                if outlier_count > 0:
                    print(f"   {col}: {outlier_count} outliers ({outlier_percentage:.1f}%)")
                    print(f"      Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
                else:
                    print(f"   {col}: No outliers detected [OK]")
        
        return outlier_info
    
    def preprocess_for_training(self, df, test_size=0.2, random_state=42):
        print("\n" + "="*60)
        print("[PROCESSING] CREATING TRAINING-READY DATASETS")
        print("="*60)
        
        feature_columns = self.categorical_columns + self.numerical_columns + self.binary_columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].copy()
        y = df[self.target_column].values
        
        print("\n[APPROACH 1] OneHot Encoding + Standard Scaling")
        preprocessor_onehot = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_columns + self.binary_columns),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_columns)
            ],
            remainder='passthrough'
        )
        
        X_onehot = preprocessor_onehot.fit_transform(X)
        
        num_feature_names = self.numerical_columns + self.binary_columns
        cat_feature_names = []
        for i, col in enumerate(self.categorical_columns):
            categories = preprocessor_onehot.named_transformers_['cat'].categories_[i][1:]
            cat_feature_names.extend([f"{col}_{cat}" for cat in categories])
        
        onehot_feature_names = num_feature_names + cat_feature_names
        
        print("[APPROACH 2] Label Encoding + Standard Scaling")
        X_label = X.copy()
        label_encoders = {}
        
        for col in self.categorical_columns:
            if col in X_label.columns:
                le = LabelEncoder()
                X_label[col] = le.fit_transform(X_label[col].astype(str))
                label_encoders[col] = le
        
        scaler_label = StandardScaler()
        X_label_scaled = scaler_label.fit_transform(X_label)
        
        X_onehot_train, X_onehot_test, y_train, y_test = train_test_split(
            X_onehot, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_label_train, X_label_test, _, _ = train_test_split(
            X_label_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"   [SUCCESS] OneHot encoded data shape: {X_onehot.shape}")
        print(f"   [SUCCESS] Label encoded data shape: {X_label_scaled.shape}")
        print(f"   [SUCCESS] Training set size: {len(X_onehot_train)}")
        print(f"   [SUCCESS] Test set size: {len(X_onehot_test)}")
        
        preprocessing_objects = {
            'onehot_preprocessor': preprocessor_onehot,
            'label_encoders': label_encoders,
            'label_scaler': scaler_label,
            'onehot_feature_names': onehot_feature_names,
            'label_feature_names': available_features
        }
        
        return {
            'onehot': {
                'X_train': X_onehot_train,
                'X_test': X_onehot_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': onehot_feature_names
            },
            'label': {
                'X_train': X_label_train,
                'X_test': X_label_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': available_features
            },
            'preprocessing_objects': preprocessing_objects
        }

    def save_processed_data(self, processed_data, output_dir='classification_processed_data'):
        print(f"\n[SAVING] Saving processed data to {output_dir}/...")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        formats_saved = []
        
        for approach in ['onehot', 'label']:
            data = processed_data[approach]
            
            np.save(f'{output_dir}/X_train_{approach}.npy', data['X_train'])
            np.save(f'{output_dir}/X_test_{approach}.npy', data['X_test'])
            np.save(f'{output_dir}/y_train.npy', data['y_train'])
            np.save(f'{output_dir}/y_test.npy', data['y_test'])
            
            json_data = {
                'X_train': data['X_train'].tolist(),
                'X_test': data['X_test'].tolist(),
                'y_train': data['y_train'].tolist(),
                'y_test': data['y_test'].tolist(),
                'feature_names': data['feature_names']
            }
            
            with open(f'{output_dir}/dataset_{approach}.json', 'w') as f:
                json.dump(json_data, f, indent=2)
            
            with open(f'{output_dir}/feature_names_{approach}.json', 'w') as f:
                json.dump(data['feature_names'], f, indent=2)
            
            formats_saved.extend([f'dataset_{approach}.json', f'feature_names_{approach}.json'])
        
        import joblib
        joblib.dump(processed_data['preprocessing_objects'], f'{output_dir}/preprocessing_objects.pkl')
        
        feature_importance_json = {}
        for key, value in self.feature_importance_scores.items():
            if key == 'chi_square':
                feature_importance_json[key] = {
                    feat: {
                        'chi2': float(data['chi2']),
                        'p_value': float(data['p_value'])
                    }
                    for feat, data in value.items()
                }
            else:
                feature_importance_json[key] = {
                    feat: float(score) for feat, score in value.items()
                }
        
        analysis_results = {
            'data_stats': self.data_stats,
            'feature_importance': feature_importance_json,
            'dataset_info': {
                'onehot_shape': list(processed_data['onehot']['X_train'].shape),
                'label_shape': list(processed_data['label']['X_train'].shape),
                'n_samples': int(len(processed_data['onehot']['y_train']) + len(processed_data['onehot']['y_test'])),
                'n_features_onehot': int(len(processed_data['onehot']['feature_names'])),
                'n_features_label': int(len(processed_data['label']['feature_names']))
            }
        }
        
        with open(f'{output_dir}/analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        formats_saved.extend(['preprocessing_objects.pkl', 'analysis_results.json'])
        
        print("   [SUCCESS] Files saved:")
        for file in formats_saved:
            print(f"      - {file}")
        
        return f'{output_dir}'
    
    def generate_preprocessing_report(self, df, processed_data, output_dir='processed_data'):
        report_path = f'{output_dir}/preprocessing_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Social Work Exam Data Preprocessing Report\n\n")
            f.write("## Dataset Overview\n\n")
            f.write(f"- **Original Shape:** {df.shape}\n")
            f.write(f"- **Features:** {len(self.categorical_columns + self.numerical_columns + self.binary_columns)}\n")
            f.write(f"- **Target:** {self.target_column}\n")
            f.write(f"- **Pass Rate:** {df[self.target_column].mean():.2%}\n\n")
            
            f.write("## Feature Categories\n\n")
            f.write(f"**Categorical Features ({len(self.categorical_columns)}):** {', '.join(self.categorical_columns)}\n\n")
            f.write(f"**Numerical Features ({len(self.numerical_columns)}):** {', '.join(self.numerical_columns)}\n\n")
            f.write(f"**Binary Features ({len(self.binary_columns)}):** {', '.join(self.binary_columns)}\n\n")
            
            f.write("## Preprocessing Approaches\n\n")
            f.write("### Approach 1: OneHot Encoding + Standard Scaling\n")
            f.write(f"- **Final Shape:** {processed_data['onehot']['X_train'].shape[1]} features\n")
            f.write(f"- **Training Samples:** {len(processed_data['onehot']['X_train'])}\n")
            f.write(f"- **Test Samples:** {len(processed_data['onehot']['X_test'])}\n\n")
            
            f.write("### Approach 2: Label Encoding + Standard Scaling\n")
            f.write(f"- **Final Shape:** {processed_data['label']['X_train'].shape[1]} features\n")
            f.write(f"- **Training Samples:** {len(processed_data['label']['X_train'])}\n")
            f.write(f"- **Test Samples:** {len(processed_data['label']['X_test'])}\n\n")
            
            if self.feature_importance_scores:
                f.write("## Feature Importance Analysis\n\n")
                
                if 'anova_f_test' in self.feature_importance_scores:
                    f.write("### ANOVA F-test Results\n")
                    for feature, score in sorted(self.feature_importance_scores['anova_f_test'].items(), 
                                               key=lambda x: x[1], reverse=True)[:10]:
                        f.write(f"- {feature}: {score:.4f}\n")
                    f.write("\n")
            
            f.write("## Data Quality Checks\n\n")
            f.write("- [OK] Missing values handled\n")
            f.write("- [OK] Outliers detected and documented\n")
            f.write("- [OK] Data split into train/test sets\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `dataset_onehot.json` - OneHot encoded dataset\n")
            f.write("- `dataset_label.json` - Label encoded dataset\n")
            f.write("- `preprocessing_objects.pkl` - Fitted preprocessing objects\n")
            f.write("- `analysis_results.json` - Complete analysis results\n")
            f.write("- `preprocessing_report.md` - This report\n\n")
            f.write("## Note\n\n")
            f.write("For correlation analysis, run `correlation_analysis.py` separately.\n")
        
        print(f"   [REPORT] Preprocessing report saved: {report_path}")

def main():
    print("[START] SOCIAL WORK EXAM DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    preprocessor = SocialWorkDataPreprocessor()
    
    csv_files_to_try = [
        'social_work_exam_dataset.csv',
        '../social_work_exam_dataset.csv',
        os.path.join('..', 'social_work_exam_dataset.csv')
    ]
    
    df = None
    for csv_file in csv_files_to_try:
        if os.path.exists(csv_file):
            print(f"[FILE] Found CSV at: {csv_file}")
            df = preprocessor.load_data(csv_file)
            break
    
    if df is None:
        print(f"[ERROR] CSV file not found in any of these locations:")
        for loc in csv_files_to_try:
            print(f"   - {loc}")
        print(f"[INFO] Current working directory: {os.getcwd()}")
        return
    
    df = preprocessor.explore_data(df)
    
    df_clean = preprocessor.handle_missing_values(df)
    
    preprocessor.feature_importance_analysis(df_clean)
    
    preprocessor.detect_outliers(df_clean)
    
    processed_data = preprocessor.preprocess_for_training(df_clean)
    
    output_dir = preprocessor.save_processed_data(processed_data)
    
    preprocessor.generate_preprocessing_report(df_clean, processed_data, output_dir)
    
    print(f"\n[COMPLETE] Preprocessing completed successfully!")
    print(f"[OUTPUT] Output directory: {output_dir}")
    print(f"[READY] Ready for training with {processed_data['onehot']['X_train'].shape[0]} training samples")
    
    if 'anova_f_test' in preprocessor.feature_importance_scores:
        print(f"\n[SUMMARY] Top 5 Most Important Features (ANOVA F-test):")
        f_scores = preprocessor.feature_importance_scores['anova_f_test']
        for i, (feature, score) in enumerate(sorted(f_scores.items(), key=lambda x: x[1], reverse=True)[:5], 1):
            print(f"   {i}. {feature}: {score:.4f}")
    
    print(f"\n[INFO] For correlation analysis, run: python correlation_analysis.py")

if __name__ == "__main__":
    main()