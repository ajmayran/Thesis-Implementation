# Ensemble Models Testing and Evaluation Report (10-Fold Cross-Validation)

## Test Dataset Information

- **Total Test Samples:** 100
- **Actual Pass Count:** 50
- **Actual Fail Count:** 50
- **Pass Rate:** 50.00%

## Ensemble Model Performance on Test Data

### STACKING LOGISTIC

**Overall Accuracy:** 0.4700

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5240
- **CV Std:** 0.0838
- **CV Min:** 0.3800
- **CV Max:** 0.6400

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            22                28
Actual Pass            25                25
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 25 (25.0%)
- **True Negatives (Correctly predicted Fail):** 22 (22.0%)
- **False Positives (Wrongly predicted Pass):** 28 (28.0%)
- **False Negatives (Wrongly predicted Fail):** 25 (25.0%)

**Classification Report:**

- **Precision (Pass class):** 0.4717
- **Recall (Pass class):** 0.5000
- **F1-Score (Pass class):** 0.4854

---

### BAGGING RANDOM FOREST

**Overall Accuracy:** 0.4500

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5380
- **CV Std:** 0.0460
- **CV Min:** 0.4600
- **CV Max:** 0.6000

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            20                30
Actual Pass            25                25
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 25 (25.0%)
- **True Negatives (Correctly predicted Fail):** 20 (20.0%)
- **False Positives (Wrongly predicted Pass):** 30 (30.0%)
- **False Negatives (Wrongly predicted Fail):** 25 (25.0%)

**Classification Report:**

- **Precision (Pass class):** 0.4545
- **Recall (Pass class):** 0.5000
- **F1-Score (Pass class):** 0.4762

---

### BOOSTING GRADIENT BOOST

**Overall Accuracy:** 0.3800

**10-Fold Cross-Validation:**

- **CV Mean:** 0.4940
- **CV Std:** 0.0820
- **CV Min:** 0.3800
- **CV Max:** 0.6800

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            11                39
Actual Pass            23                27
```

**Detailed Metrics:**

- **True Positives (Correctly predicted Pass):** 27 (27.0%)
- **True Negatives (Correctly predicted Fail):** 11 (11.0%)
- **False Positives (Wrongly predicted Pass):** 39 (39.0%)
- **False Negatives (Wrongly predicted Fail):** 23 (23.0%)

**Classification Report:**

- **Precision (Pass class):** 0.4091
- **Recall (Pass class):** 0.5400
- **F1-Score (Pass class):** 0.4655

---

## 10-Fold Cross-Validation Explained

10-fold cross-validation provides a robust estimate of model performance by:

1. Splitting the dataset into 10 equal parts (folds)
2. Training on 9 folds and testing on 1 fold
3. Repeating this process 10 times (each fold serves as test set once)
4. Averaging the results for final performance metrics

This approach reduces variance and provides more reliable performance estimates.

## Ensemble Techniques Used

### 1. Bagging (Random Forest)
- Multiple decision trees trained on bootstrap samples
- Reduces variance and prevents overfitting

### 2. Boosting (Gradient Boosting)
- Sequential training where each model corrects previous errors
- Reduces bias and improves accuracy

### 3. Stacking (with Logistic Regression)
- Combines predictions from 3 base models (KNN, Decision Tree, Random Forest)
- Meta-learner (Logistic Regression) learns optimal combination

## How to Interpret Results

- **True Positive:** Model correctly predicted student will PASS
- **True Negative:** Model correctly predicted student will FAIL
- **False Positive:** Model predicted PASS but student actually FAILED (Type I Error)
- **False Negative:** Model predicted FAIL but student actually PASSED (Type II Error)

## Testing Process

1. Ensemble models were validated using 10-fold cross-validation
2. Models were trained on 80% of data (training set)
3. Models make predictions on unseen 20% (test set)
4. Predictions are compared with actual results
5. Accuracy and other metrics are calculated
