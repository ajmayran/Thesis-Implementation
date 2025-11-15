# Model Testing and Evaluation Report (10-Fold Cross-Validation)

## Test Dataset Information

- **Total Test Samples:** 110
- **Actual Pass Count:** 55
- **Actual Fail Count:** 55
- **Pass Rate:** 50.00%

## Model Performance on Test Data

### NEURAL_NETWORK

**Overall Accuracy:** 0.9727

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5695
- **CV Std:** 0.0384
- **CV Min:** 0.5091
- **CV Max:** 0.6296

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            53                 2
Actual Pass             1                54
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 54 (49.1%)
- **True Negatives (Correctly predicted Fail):** 53 (48.2%)
- **False Positives (Wrongly predicted Pass):** 2 (1.8%)
- **False Negatives (Wrongly predicted Fail):** 1 (0.9%)

**Classification Report:**

- **Precision (Pass class):** 0.9643
- **Recall (Pass class):** 0.9818
- **F1-Score (Pass class):** 0.9730

**Best Hyperparameters:** {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50,)}

---

### RANDOM_FOREST

**Overall Accuracy:** 0.7455

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5533
- **CV Std:** 0.0344
- **CV Min:** 0.5091
- **CV Max:** 0.6111

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            46                 9
Actual Pass            19                36
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 36 (32.7%)
- **True Negatives (Correctly predicted Fail):** 46 (41.8%)
- **False Positives (Wrongly predicted Pass):** 9 (8.2%)
- **False Negatives (Wrongly predicted Fail):** 19 (17.3%)

**Classification Report:**

- **Precision (Pass class):** 0.8000
- **Recall (Pass class):** 0.6545
- **F1-Score (Pass class):** 0.7200

**Best Hyperparameters:** {'max_depth': 5, 'n_estimators': 100}

---

### SVM

**Overall Accuracy:** 0.7455

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5606
- **CV Std:** 0.0566
- **CV Min:** 0.4182
- **CV Max:** 0.6296

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            43                12
Actual Pass            16                39
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 39 (35.5%)
- **True Negatives (Correctly predicted Fail):** 43 (39.1%)
- **False Positives (Wrongly predicted Pass):** 12 (10.9%)
- **False Negatives (Wrongly predicted Fail):** 16 (14.5%)

**Classification Report:**

- **Precision (Pass class):** 0.7647
- **Recall (Pass class):** 0.7091
- **F1-Score (Pass class):** 0.7358

**Best Hyperparameters:** {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}

---

### KNN

**Overall Accuracy:** 0.7182

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5603
- **CV Std:** 0.0625
- **CV Min:** 0.4444
- **CV Max:** 0.6545

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            43                12
Actual Pass            19                36
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 36 (32.7%)
- **True Negatives (Correctly predicted Fail):** 43 (39.1%)
- **False Positives (Wrongly predicted Pass):** 12 (10.9%)
- **False Negatives (Wrongly predicted Fail):** 19 (17.3%)

**Classification Report:**

- **Precision (Pass class):** 0.7500
- **Recall (Pass class):** 0.6545
- **F1-Score (Pass class):** 0.6990

**Best Hyperparameters:** {'n_neighbors': 5, 'weights': 'uniform'}

---

### DECISION_TREE

**Overall Accuracy:** 0.5455

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5459
- **CV Std:** 0.0556
- **CV Min:** 0.4727
- **CV Max:** 0.6667

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            49                 6
Actual Pass            44                11
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 11 (10.0%)
- **True Negatives (Correctly predicted Fail):** 49 (44.5%)
- **False Positives (Wrongly predicted Pass):** 6 (5.5%)
- **False Negatives (Wrongly predicted Fail):** 44 (40.0%)

**Classification Report:**

- **Precision (Pass class):** 0.6471
- **Recall (Pass class):** 0.2000
- **F1-Score (Pass class):** 0.3056

**Best Hyperparameters:** {'max_depth': 3, 'min_samples_split': 2}

---

### NAIVE_BAYES

**Overall Accuracy:** 0.4909

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5316
- **CV Std:** 0.0803
- **CV Min:** 0.4444
- **CV Max:** 0.7037

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            35                20
Actual Pass            36                19
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 19 (17.3%)
- **True Negatives (Correctly predicted Fail):** 35 (31.8%)
- **False Positives (Wrongly predicted Pass):** 20 (18.2%)
- **False Negatives (Wrongly predicted Fail):** 36 (32.7%)

**Classification Report:**

- **Precision (Pass class):** 0.4872
- **Recall (Pass class):** 0.3455
- **F1-Score (Pass class):** 0.4043

**Best Hyperparameters:** {'var_smoothing': 1e-09}

---

## 10-Fold Cross-Validation Explained

10-fold cross-validation provides a robust estimate of model performance by:

1. Splitting the dataset into 10 equal parts (folds)
2. Training on 9 folds and testing on 1 fold
3. Repeating this process 10 times (each fold serves as test set once)
4. Averaging the results for final performance metrics

This approach reduces variance and provides more reliable performance estimates.

## How to Interpret Results

- **True Positive:** Model correctly predicted student will PASS
- **True Negative:** Model correctly predicted student will FAIL
- **False Positive:** Model predicted PASS but student actually FAILED (Type I Error)
- **False Negative:** Model predicted FAIL but student actually PASSED (Type II Error)

## Testing Process

1. Models were trained using 10-fold cross-validation
2. Best hyperparameters were selected based on CV performance
3. Final models make predictions on held-out test set
4. Performance metrics are calculated and compared
