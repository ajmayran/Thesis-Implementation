# Social Work Exam Data Preprocessing Report

## Dataset Overview

- **Original Shape:** (546, 15)
- **Features:** 13
- **Target:** Passed
- **Pass Rate:** 50.37%

## Feature Categories

**Categorical Features (3):** Gender, IncomeLevel, EmploymentStatus

**Numerical Features (8):** Age, StudyHours, SleepHours, Confidence, MockExamScore, GPA, InternshipGrade, TestAnxiety

**Binary Features (2):** ReviewCenter, Scholarship

## Preprocessing Approaches

### Approach 1: OneHot Encoding + Standard Scaling
- **Final Shape:** 16 features
- **Training Samples:** 436
- **Test Samples:** 110

### Approach 2: Label Encoding + Standard Scaling
- **Final Shape:** 13 features
- **Training Samples:** 436
- **Test Samples:** 110

## Feature Importance Analysis

### ANOVA F-test Results
- MockExamScore: 10.2600
- GPA: 6.7583
- StudyHours: 3.7202
- Gender: 2.9335
- TestAnxiety: 2.8672
- EmploymentStatus: 2.8265
- InternshipGrade: 1.2623
- Scholarship: 1.2167
- Confidence: 0.6762
- Age: 0.4606

### Pearson Correlation with Target
- MockExamScore: r=0.1820, p=0.0013 (significant)
- GPA: r=0.1108, p=0.0096 (significant)
- StudyHours: r=0.0824, p=0.0543
- Gender: r=-0.0732, p=0.0873
- TestAnxiety: r=0.0724, p=0.0910
- EmploymentStatus: r=-0.0719, p=0.0933
- InternshipGrade: r=0.0481, p=0.2617
- Scholarship: r=0.0472, p=0.2705
- Confidence: r=0.0352, p=0.4113
- Age: r=0.0291, p=0.4976

## Data Quality Checks

- [OK] Missing values handled
- [OK] Outliers detected and documented
- [OK] Feature correlations analyzed
- [OK] Pearson correlation computed
- [OK] Data split into train/test sets

## Files Generated

- `dataset_onehot.json` - OneHot encoded dataset
- `dataset_label.json` - Label encoded dataset
- `preprocessing_objects.pkl` - Fitted preprocessing objects
- `analysis_results.json` - Complete analysis results
- `preprocessing_report.md` - This report
