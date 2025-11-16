# Ensemble Models Testing Report

## Test Dataset

Total Test Samples: 110
Pass Count: 55
Fail Count: 55

## Ensemble Performance

### BAGGING

**Accuracy:** 0.5545

**10-Fold CV Mean:** 0.5480
**10-Fold CV Std:** 0.0558

**Confusion Matrix:**

```
TN: 32  FP: 23
FN: 26  TP: 29
```

**Precision:** 0.5577
**Recall:** 0.5273
**F1-Score:** 0.5421

---

### STACKING

**Accuracy:** 0.5182

**10-Fold CV Mean:** 0.5551
**10-Fold CV Std:** 0.0641

**Confusion Matrix:**

```
TN: 34  FP: 21
FN: 32  TP: 23
```

**Precision:** 0.5227
**Recall:** 0.4182
**F1-Score:** 0.4646

---

### BOOSTING

**Accuracy:** 0.5000

**10-Fold CV Mean:** 0.4999
**10-Fold CV Std:** 0.0613

**Confusion Matrix:**

```
TN: 31  FP: 24
FN: 31  TP: 24
```

**Precision:** 0.5000
**Recall:** 0.4364
**F1-Score:** 0.4660

---

## Ensemble Methods

**Bagging:** 10 Random Forest models with bootstrap sampling

**Boosting:** Gradient Boosting with 100 estimators

**Stacking:** Combines 6 fresh base model instances (KNN, Decision Tree, Random Forest, SVM, Neural Network, Naive Bayes) with Logistic Regression meta-learner

