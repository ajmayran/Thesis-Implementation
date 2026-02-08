# Model Testing and Evaluation Report (10-Fold Cross-Validation)

## Test Dataset Information

- **Total Test Samples:** 110
- **Actual Pass Count:** 55
- **Actual Fail Count:** 55
- **Pass Rate:** 50.00%

## Model Performance on Test Data

### RANDOM_FOREST

**Overall Accuracy:** 0.5182

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5802
- **CV Std:** 0.0569
- **CV Min:** 0.4651
- **CV Max:** 0.6591

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            32                23
Actual Pass            30                25
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 25 (22.7%)
- **True Negatives (Correctly predicted Fail):** 32 (29.1%)
- **False Positives (Wrongly predicted Pass):** 23 (20.9%)
- **False Negatives (Wrongly predicted Fail):** 30 (27.3%)

**Classification Report:**

- **Precision (Pass class):** 0.5208
- **Recall (Pass class):** 0.4545
- **F1-Score (Pass class):** 0.4854

**Best Hyperparameters:** {'max_depth': 15, 'n_estimators': 100}

---

### DECISION_TREE

**Overall Accuracy:** 0.5091

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5826
- **CV Std:** 0.0627
- **CV Min:** 0.4545
- **CV Max:** 0.6591

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            31                24
Actual Pass            30                25
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 25 (22.7%)
- **True Negatives (Correctly predicted Fail):** 31 (28.2%)
- **False Positives (Wrongly predicted Pass):** 24 (21.8%)
- **False Negatives (Wrongly predicted Fail):** 30 (27.3%)

**Classification Report:**

- **Precision (Pass class):** 0.5102
- **Recall (Pass class):** 0.4545
- **F1-Score (Pass class):** 0.4808

**Best Hyperparameters:** {'max_depth': 10, 'min_samples_split': 5}

---

### SVM

**Overall Accuracy:** 0.5000

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5874
- **CV Std:** 0.0650
- **CV Min:** 0.4545
- **CV Max:** 0.6744

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            34                21
Actual Pass            34                21
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 21 (19.1%)
- **True Negatives (Correctly predicted Fail):** 34 (30.9%)
- **False Positives (Wrongly predicted Pass):** 21 (19.1%)
- **False Negatives (Wrongly predicted Fail):** 34 (30.9%)

**Classification Report:**

- **Precision (Pass class):** 0.5000
- **Recall (Pass class):** 0.3818
- **F1-Score (Pass class):** 0.4330

**Best Hyperparameters:** {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}

---

### NEURAL_NETWORK

**Overall Accuracy:** 0.4909

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5733
- **CV Std:** 0.0506
- **CV Min:** 0.4773
- **CV Max:** 0.6818

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            32                23
Actual Pass            33                22
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 22 (20.0%)
- **True Negatives (Correctly predicted Fail):** 32 (29.1%)
- **False Positives (Wrongly predicted Pass):** 23 (20.9%)
- **False Negatives (Wrongly predicted Fail):** 33 (30.0%)

**Classification Report:**

- **Precision (Pass class):** 0.4889
- **Recall (Pass class):** 0.4000
- **F1-Score (Pass class):** 0.4400

**Best Hyperparameters:** {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (50,), 'learning_rate_init': 0.01}

---

### NAIVE_BAYES

**Overall Accuracy:** 0.4727

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5549
- **CV Std:** 0.0569
- **CV Min:** 0.4419
- **CV Max:** 0.6364

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            34                21
Actual Pass            37                18
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 18 (16.4%)
- **True Negatives (Correctly predicted Fail):** 34 (30.9%)
- **False Positives (Wrongly predicted Pass):** 21 (19.1%)
- **False Negatives (Wrongly predicted Fail):** 37 (33.6%)

**Classification Report:**

- **Precision (Pass class):** 0.4615
- **Recall (Pass class):** 0.3273
- **F1-Score (Pass class):** 0.3830

**Best Hyperparameters:** {'var_smoothing': 1e-09}

---

### KNN

**Overall Accuracy:** 0.4636

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5736
- **CV Std:** 0.0600
- **CV Min:** 0.5000
- **CV Max:** 0.6744

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            33                22
Actual Pass            37                18
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 18 (16.4%)
- **True Negatives (Correctly predicted Fail):** 33 (30.0%)
- **False Positives (Wrongly predicted Pass):** 22 (20.0%)
- **False Negatives (Wrongly predicted Fail):** 37 (33.6%)

**Classification Report:**

- **Precision (Pass class):** 0.4500
- **Recall (Pass class):** 0.3273
- **F1-Score (Pass class):** 0.3789

**Best Hyperparameters:** {'n_neighbors': 7, 'weights': 'uniform'}

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
