# Model Testing and Evaluation Report (10-Fold Cross-Validation)

## Test Dataset Information

- **Total Test Samples:** 100
- **Actual Pass Count:** 50
- **Actual Fail Count:** 50
- **Pass Rate:** 50.00%

## Model Performance on Test Data

### RANDOM_FOREST

**Overall Accuracy:** 0.8300

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5420
- **CV Std:** 0.0672
- **CV Min:** 0.4600
- **CV Max:** 0.6600

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            38                12
Actual Pass             5                45
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 45 (45.0%)
- **True Negatives (Correctly predicted Fail):** 38 (38.0%)
- **False Positives (Wrongly predicted Pass):** 12 (12.0%)
- **False Negatives (Wrongly predicted Fail):** 5 (5.0%)

**Classification Report:**

- **Precision (Pass class):** 0.7895
- **Recall (Pass class):** 0.9000
- **F1-Score (Pass class):** 0.8411

**Best Hyperparameters:** {'max_depth': 5, 'n_estimators': 50}

---

### DECISION_TREE

**Overall Accuracy:** 0.6900

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5360
- **CV Std:** 0.0571
- **CV Min:** 0.4200
- **CV Max:** 0.6200

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            32                18
Actual Pass            13                37
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 37 (37.0%)
- **True Negatives (Correctly predicted Fail):** 32 (32.0%)
- **False Positives (Wrongly predicted Pass):** 18 (18.0%)
- **False Negatives (Wrongly predicted Fail):** 13 (13.0%)

**Classification Report:**

- **Precision (Pass class):** 0.6727
- **Recall (Pass class):** 0.7400
- **F1-Score (Pass class):** 0.7048

**Best Hyperparameters:** {'max_depth': 5, 'min_samples_split': 5}

---

### KNN

**Overall Accuracy:** 0.5500

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5400
- **CV Std:** 0.0764
- **CV Min:** 0.4200
- **CV Max:** 0.6600

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            27                23
Actual Pass            22                28
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 28 (28.0%)
- **True Negatives (Correctly predicted Fail):** 27 (27.0%)
- **False Positives (Wrongly predicted Pass):** 23 (23.0%)
- **False Negatives (Wrongly predicted Fail):** 22 (22.0%)

**Classification Report:**

- **Precision (Pass class):** 0.5490
- **Recall (Pass class):** 0.5600
- **F1-Score (Pass class):** 0.5545

**Best Hyperparameters:** {'n_neighbors': 9, 'weights': 'uniform'}

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
