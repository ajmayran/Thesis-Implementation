import numpy as np
import pandas as pd

class RegressionPreprocessor:
    """
    Preprocessing pipeline for regression models
    Handles imputation, encoding, and scaling
    """
    
    def __init__(self, iterative_imputer, median_imputer, label_encoders, scaler, imputation_config):
        """
        Initialize preprocessor with fitted transformers
        
        Args:
            iterative_imputer: Fitted IterativeImputer for grade columns
            median_imputer: Fitted SimpleImputer for score columns
            label_encoders: Dictionary of fitted LabelEncoders
            scaler: Fitted StandardScaler
            imputation_config: Dictionary specifying which columns use which imputer
        """
        self.iterative_imputer = iterative_imputer
        self.median_imputer = median_imputer
        self.label_encoders = label_encoders
        self.scaler = scaler
        self.imputation_config = imputation_config
        
        self.categorical_columns = list(label_encoders.keys())
        self.grade_columns = imputation_config['grade_columns']
        self.score_columns = imputation_config['score_columns']
    
    def transform(self, X):
        """
        Transform input data through preprocessing pipeline
        
        Args:
            X: DataFrame with raw features
            
        Returns:
            Preprocessed numpy array ready for model prediction
        """
        X = X.copy()
        
        # Step 1: Impute grade columns with IterativeImputer
        if any(X[self.grade_columns].isnull().any()):
            X[self.grade_columns] = self.iterative_imputer.transform(X[self.grade_columns])
        
        # Step 2: Impute score columns with median imputer
        if any(X[self.score_columns].isnull().any()):
            X[self.score_columns] = self.median_imputer.transform(X[self.score_columns])
        
        # Step 3: Encode categorical variables
        for col in self.categorical_columns:
            if col in X.columns:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Step 4: Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def get_feature_names(self):
        """Return list of feature names in correct order"""
        return self.categorical_columns + self.grade_columns + self.score_columns