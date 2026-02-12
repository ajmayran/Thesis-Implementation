# Preprocessed Data for Regression Models

## Data Split
- Training samples: 88
- Test samples: 22
- Test size: 20%
- Random state: 42

## Features
- Total features: 16
- Categorical: 3
- Numerical: 12
- Binary: 1

## Preprocessing Steps
1. Train-test split (prevents data leakage)
2. Missing value imputation (fit on train only)
   - Grade columns: IterativeImputer
   - Score columns: Median imputation
3. Categorical encoding (fit on train only)
   - LabelEncoder for categorical variables
4. Feature scaling (fit on train only)
   - StandardScaler (mean=0, std=1)

## Files
- X_train.npy: Training features (scaled)
- X_test.npy: Test features (scaled)
- y_train.npy: Training target
- y_test.npy: Test target
- scaler.pkl: Fitted StandardScaler
- label_encoders.pkl: Dictionary of fitted LabelEncoders
- iterative_imputer.pkl: Fitted IterativeImputer
- median_imputer.pkl: Fitted SimpleImputer
- feature_names.json: List of feature names
- preprocessing_config.json: Complete preprocessing configuration
- imputation_config.json: Imputation strategy configuration
- feature_importance_analysis.json: Feature importance results

## Data Validation
All data has been validated for:
- No missing values (NaN)
- No infinite values (Inf)
- Consistent shapes
- Proper scaling (mean ~0, std ~1 for training data)

Generated: 2026-02-11 23:41:24.202184
