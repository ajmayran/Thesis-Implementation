# Social Work Exam Data Preprocessing Report

## Dataset Overview

- **Original Shape:** (500, 14)
- **Features:** 12
- **Target:** Passed
- **Pass Rate:** 50.00%

## Feature Categories

**Categorical Features (3):** Gender, IncomeLevel, EmploymentStatus

**Numerical Features (8):** Age, StudyHours, SleepHours, Confidence, MockExamScore, GPA, Scholarship, InternshipGrade

**Binary Features (1):** ReviewCenter

## Preprocessing Approaches

### Approach 1: OneHot Encoding + Standard Scaling
- **Final Shape:** 15 features
- **Training Samples:** 400
- **Test Samples:** 100

### Approach 2: Label Encoding + Standard Scaling
- **Final Shape:** 12 features
- **Training Samples:** 400
- **Test Samples:** 100

## Feature Importance Analysis

### ANOVA F-test Results
- Age: 13.8633
- Gender: 3.2077
- StudyHours: 2.0764
- GPA: 1.4949
- ReviewCenter: 1.3551
- MockExamScore: 1.2161
- Confidence: 1.1838
- SleepHours: 0.8548
- EmploymentStatus: 0.8168
- InternshipGrade: 0.4784

## Data Quality Checks

- [OK] Missing values handled
- [OK] Outliers detected and documented
- [OK] Feature correlations analyzed
- [OK] Data split into train/test sets

## Files Generated

- `dataset_onehot.json` - OneHot encoded dataset
- `dataset_label.json` - Label encoded dataset
- `preprocessing_objects.pkl` - Fitted preprocessing objects
- `analysis_results.json` - Complete analysis results
- `preprocessing_report.md` - This report
