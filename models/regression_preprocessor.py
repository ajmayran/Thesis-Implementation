from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

class RegressionPreprocessor:
    """Custom preprocessor for regression models"""
    
    def __init__(self, imputer, label_encoders, scaler):
        self.imputer = imputer
        self.label_encoders = label_encoders
        self.scaler = scaler
        self.categorical_columns = ['Gender', 'IncomeLevel', 'EmploymentStatus']
        self.numerical_columns = ['Age', 'StudyHours', 'SleepHours', 'Confidence', 'TestAnxiety',
                                'MockExamScore', 'GPA', 'Scholarship', 'InternshipGrade']
        self.binary_columns = ['ReviewCenter']
    
    def transform(self, X):
        """Transform input data matching training pipeline"""
        X = X.copy()
        
        # Step 1: Impute missing values
        X[self.numerical_columns] = self.imputer.transform(X[self.numerical_columns])
        
        # Step 2: Label encode categorical
        for col in self.categorical_columns:
            if col in X.columns:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Step 3: Scale all features
        return self.scaler.transform(X)