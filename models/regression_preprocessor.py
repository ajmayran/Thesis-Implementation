import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class RegressionPreprocessor:
    """
    Preprocessor for regression models that handles imputation, encoding, and scaling.
    Designed to prevent data leakage by fitting only on training data.
    """
    
    def __init__(self, iterative_imputer=None, median_imputer=None, 
                 label_encoders=None, scaler=None, imputation_config=None):
        """
        Initialize preprocessor with fitted transformers.
        
        Args:
            iterative_imputer: Fitted IterativeImputer for grade columns
            median_imputer: Fitted SimpleImputer for score columns
            label_encoders: Dictionary of fitted LabelEncoders
            scaler: Fitted StandardScaler
            imputation_config: Dictionary with column groupings
        """
        self.iterative_imputer = iterative_imputer
        self.median_imputer = median_imputer
        self.label_encoders = label_encoders or {}
        self.scaler = scaler
        
        # Extract column configurations from imputation_config
        if imputation_config:
            self.grade_columns = imputation_config.get('grade_columns', [])
            self.score_columns = imputation_config.get('score_columns', [])
            self.categorical_columns = imputation_config.get('categorical_columns', [])
            self.numerical_columns = imputation_config.get('numerical_columns', [])
            self.binary_columns = imputation_config.get('binary_columns', [])
        else:
            self.grade_columns = []
            self.score_columns = []
            self.categorical_columns = []
            self.numerical_columns = []
            self.binary_columns = []
        
        self.feature_names = (self.categorical_columns + 
                            self.numerical_columns + 
                            self.binary_columns)
    
    def transform(self, X):
        """
        Transform input data using fitted preprocessors.
        
        Args:
            X: DataFrame or dict with raw feature values
            
        Returns:
            Scaled numpy array ready for model prediction
        """
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a DataFrame or dictionary")
        
        # Create copy to avoid modifying original
        X_processed = X.copy()
        
        # Step 1: Handle missing values
        if self.iterative_imputer and self.grade_columns:
            grade_cols_present = [col for col in self.grade_columns if col in X_processed.columns]
            if grade_cols_present and X_processed[grade_cols_present].isnull().any().any():
                X_processed[grade_cols_present] = self.iterative_imputer.transform(
                    X_processed[grade_cols_present]
                )
        
        if self.median_imputer and self.score_columns:
            score_cols_present = [col for col in self.score_columns if col in X_processed.columns]
            if score_cols_present and X_processed[score_cols_present].isnull().any().any():
                X_processed[score_cols_present] = self.median_imputer.transform(
                    X_processed[score_cols_present]
                )
        
        # Step 2: Encode categorical variables
        for col in self.categorical_columns:
            if col in self.label_encoders and col in X_processed.columns:
                le = self.label_encoders[col]
                
                def safe_transform(x):
                    try:
                        return le.transform([str(x)])[0]
                    except (ValueError, KeyError):
                        # Return first class if unseen category
                        return le.transform([le.classes_[0]])[0]
                
                X_processed[col] = X_processed[col].apply(safe_transform)
        
        # Step 3: Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X_processed[self.feature_names])
        else:
            X_scaled = X_processed[self.feature_names].values
        
        return X_scaled
    
    def get_feature_names(self):
        """Return list of feature names in order"""
        return self.feature_names
    
    def __repr__(self):
        return (f"RegressionPreprocessor("
                f"n_features={len(self.feature_names)}, "
                f"n_categorical={len(self.categorical_columns)}, "
                f"n_numerical={len(self.numerical_columns)}, "
                f"n_binary={len(self.binary_columns)})")