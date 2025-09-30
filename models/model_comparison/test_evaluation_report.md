# Model Testing and Evaluation Report

## Test Dataset Information

- **Total Test Samples:** 100
- **Actual Pass Count:** 50
- **Actual Fail Count:** 50
- **Pass Rate:** 50.00%

## Model Performance on Test Data

### DECISION_TREE

**Overall Accuracy:** 0.4700

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            21                29
Actual Pass            24                26
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 26 (26.0%)
- **True Negatives (Correctly predicted Fail):** 21 (21.0%)
- **False Positives (Wrongly predicted Pass):** 29 (29.0%)
- **False Negatives (Wrongly predicted Fail):** 24 (24.0%)

**Classification Report:**

- **Precision (Pass class):** 0.4727
- **Recall (Pass class):** 0.5200
- **F1-Score (Pass class):** 0.4952

**Best Hyperparameters:** {'max_depth': 7, 'min_samples_split': 10}

---

### KNN

**Overall Accuracy:** 0.4600

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            23                27
Actual Pass            27                23
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 23 (23.0%)
- **True Negatives (Correctly predicted Fail):** 23 (23.0%)
- **False Positives (Wrongly predicted Pass):** 27 (27.0%)
- **False Negatives (Wrongly predicted Fail):** 27 (27.0%)

**Classification Report:**

- **Precision (Pass class):** 0.4600
- **Recall (Pass class):** 0.4600
- **F1-Score (Pass class):** 0.4600

**Best Hyperparameters:** {'n_neighbors': 7, 'weights': 'uniform'}

---

### RANDOM_FOREST

**Overall Accuracy:** 0.4300

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            22                28
Actual Pass            29                21
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 21 (21.0%)
- **True Negatives (Correctly predicted Fail):** 22 (22.0%)
- **False Positives (Wrongly predicted Pass):** 28 (28.0%)
- **False Negatives (Wrongly predicted Fail):** 29 (29.0%)

**Classification Report:**

- **Precision (Pass class):** 0.4286
- **Recall (Pass class):** 0.4200
- **F1-Score (Pass class):** 0.4242

**Best Hyperparameters:** {'max_depth': 10, 'n_estimators': 50}

---

## How to Interpret Results

- **True Positive:** Model correctly predicted student will PASS
- **True Negative:** Model correctly predicted student will FAIL
- **False Positive:** Model predicted PASS but student actually FAILED (Type I Error)
- **False Negative:** Model predicted FAIL but student actually PASSED (Type II Error)

## Testing Process

1. Model was trained on 80% of data (training set)
2. Model makes predictions on unseen 20% (test set)
3. Predictions are compared with actual results
4. Accuracy and other metrics are calculated
