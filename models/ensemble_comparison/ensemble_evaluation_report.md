# Ensemble Models Testing Report (10-Fold Cross-Validation)

## Test Dataset Information

Total Test Samples: 110
Pass Count: 55
Fail Count: 55
Pass Rate: 50.00%

## Ensemble Performance on Test Data

### STACKING

**Overall Accuracy:** 0.5273

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5599
- **CV Std:** 0.0488
- **CV Min:** 0.5000
- **CV Max:** 0.6744

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            32                23
Actual Pass            29                26
```

**Detailed Metrics:**

- **True Positives:** 26 (23.6%)
- **True Negatives:** 32 (29.1%)
- **False Positives:** 23 (20.9%)
- **False Negatives:** 29 (26.4%)

**Classification Report:**

- **Precision (Pass class):** 0.5306
- **Recall (Pass class):** 0.4727
- **F1-Score (Pass class):** 0.5000

---

### BAGGING

**Overall Accuracy:** 0.5182

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5779
- **CV Std:** 0.0536
- **CV Min:** 0.4884
- **CV Max:** 0.6364

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            29                26
Actual Pass            27                28
```

**Detailed Metrics:**

- **True Positives:** 28 (25.5%)
- **True Negatives:** 29 (26.4%)
- **False Positives:** 26 (23.6%)
- **False Negatives:** 27 (24.5%)

**Classification Report:**

- **Precision (Pass class):** 0.5185
- **Recall (Pass class):** 0.5091
- **F1-Score (Pass class):** 0.5138

---

### WEIGHTED_VOTING

**Overall Accuracy:** 0.5000

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5804
- **CV Std:** 0.0446
- **CV Min:** 0.5227
- **CV Max:** 0.6512

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            31                24
Actual Pass            31                24
```

**Detailed Metrics:**

- **True Positives:** 24 (21.8%)
- **True Negatives:** 31 (28.2%)
- **False Positives:** 24 (21.8%)
- **False Negatives:** 31 (28.2%)

**Classification Report:**

- **Precision (Pass class):** 0.5000
- **Recall (Pass class):** 0.4364
- **F1-Score (Pass class):** 0.4660

---

### BOOSTING

**Overall Accuracy:** 0.4909

**10-Fold Cross-Validation:**

- **CV Mean:** 0.5801
- **CV Std:** 0.0579
- **CV Min:** 0.4884
- **CV Max:** 0.6818

**Confusion Matrix:**

```
                Predicted Fail    Predicted Pass
Actual Fail            30                25
Actual Pass            31                24
```

**Detailed Metrics:**

- **True Positives:** 24 (21.8%)
- **True Negatives:** 30 (27.3%)
- **False Positives:** 25 (22.7%)
- **False Negatives:** 31 (28.2%)

**Classification Report:**

- **Precision (Pass class):** 0.4898
- **Recall (Pass class):** 0.4364
- **F1-Score (Pass class):** 0.4615

---

## Ensemble Methods Explained

**Bagging:** 10 Random Forest models with bootstrap sampling

**Boosting:** Gradient Boosting with 100 estimators

**Stacking:** Combines 6 fresh base model instances (KNN, Decision Tree, Random Forest, SVM, Neural Network, Naive Bayes) with Logistic Regression meta-learner

## 10-Fold Cross-Validation

Cross-validation performed on training set only to prevent data leakage.
Test set kept completely separate for final evaluation.
