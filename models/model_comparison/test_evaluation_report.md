# Model Testing and Evaluation Report (10-Fold Cross-Validation)

## Test Dataset Information

- **Total Test Samples:** 110
- **Actual Pass Count:** 55
- **Actual Fail Count:** 55
- **Pass Rate:** 50.00%

## Model Performance on Test Data

### NEURAL_NETWORK

**Overall Accuracy:** 1.0000

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5716
- **CV Std:** 0.0757
- **CV Min:** 0.4000
- **CV Max:** 0.6727

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            55                 0
Actual Pass             0                55
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 55 (50.0%)
- **True Negatives (Correctly predicted Fail):** 55 (50.0%)
- **False Positives (Wrongly predicted Pass):** 0 (0.0%)
- **False Negatives (Wrongly predicted Fail):** 0 (0.0%)

**Classification Report:**

- **Precision (Pass class):** 1.0000
- **Recall (Pass class):** 1.0000
- **F1-Score (Pass class):** 1.0000

**Best Hyperparameters:** {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (100,)}

---

### RANDOM_FOREST

**Overall Accuracy:** 0.7636

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5552
- **CV Std:** 0.0577
- **CV Min:** 0.4630
- **CV Max:** 0.6481

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            46                 9
Actual Pass            17                38
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 38 (34.5%)
- **True Negatives (Correctly predicted Fail):** 46 (41.8%)
- **False Positives (Wrongly predicted Pass):** 9 (8.2%)
- **False Negatives (Wrongly predicted Fail):** 17 (15.5%)

**Classification Report:**

- **Precision (Pass class):** 0.8085
- **Recall (Pass class):** 0.6909
- **F1-Score (Pass class):** 0.7451

**Best Hyperparameters:** {'max_depth': 5, 'n_estimators': 100}

---

### SVM

**Overall Accuracy:** 0.7636

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5569
- **CV Std:** 0.0778
- **CV Min:** 0.4000
- **CV Max:** 0.6364

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            43                12
Actual Pass            14                41
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 41 (37.3%)
- **True Negatives (Correctly predicted Fail):** 43 (39.1%)
- **False Positives (Wrongly predicted Pass):** 12 (10.9%)
- **False Negatives (Wrongly predicted Fail):** 14 (12.7%)

**Classification Report:**

- **Precision (Pass class):** 0.7736
- **Recall (Pass class):** 0.7455
- **F1-Score (Pass class):** 0.7593

**Best Hyperparameters:** {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}

---

### DECISION_TREE

**Overall Accuracy:** 0.7364

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5422
- **CV Std:** 0.0521
- **CV Min:** 0.4630
- **CV Max:** 0.6182

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            46                 9
Actual Pass            20                35
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 35 (31.8%)
- **True Negatives (Correctly predicted Fail):** 46 (41.8%)
- **False Positives (Wrongly predicted Pass):** 9 (8.2%)
- **False Negatives (Wrongly predicted Fail):** 20 (18.2%)

**Classification Report:**

- **Precision (Pass class):** 0.7955
- **Recall (Pass class):** 0.6364
- **F1-Score (Pass class):** 0.7071

**Best Hyperparameters:** {'max_depth': 10, 'min_samples_split': 10}

---

### KNN

**Overall Accuracy:** 0.6909

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5438
- **CV Std:** 0.0697
- **CV Min:** 0.4074
- **CV Max:** 0.6545

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            48                 7
Actual Pass            27                28
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 28 (25.5%)
- **True Negatives (Correctly predicted Fail):** 48 (43.6%)
- **False Positives (Wrongly predicted Pass):** 7 (6.4%)
- **False Negatives (Wrongly predicted Fail):** 27 (24.5%)

**Classification Report:**

- **Precision (Pass class):** 0.8000
- **Recall (Pass class):** 0.5091
- **F1-Score (Pass class):** 0.6222

**Best Hyperparameters:** {'n_neighbors': 3, 'weights': 'uniform'}

---

### NAIVE_BAYES

**Overall Accuracy:** 0.4727

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5224
- **CV Std:** 0.0613
- **CV Min:** 0.4182
- **CV Max:** 0.6296

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            35                20
Actual Pass            38                17
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 17 (15.5%)
- **True Negatives (Correctly predicted Fail):** 35 (31.8%)
- **False Positives (Wrongly predicted Pass):** 20 (18.2%)
- **False Negatives (Wrongly predicted Fail):** 38 (34.5%)

**Classification Report:**

- **Precision (Pass class):** 0.4595
- **Recall (Pass class):** 0.3091
- **F1-Score (Pass class):** 0.3696

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
