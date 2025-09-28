# Social Work Exam Data Preprocessing Report

## Dataset Overview

- **Original Shape:** (500, 14)
- **Features:** 13
- **Target:** Passed
- **Pass Rate:** 50.00%

## Feature Categories

**Categorical Features (3):** Gender, IncomeLevel, EmploymentStatus

**Numerical Features (9):** Age, StudyHours, SleepHours, Confidence, MockExamScore, GPA, Scholarship, InternshipGrade, ExamResultPercent

**Binary Features (1):** ReviewCenter

## Preprocessing Approaches

### Approach 1: OneHot Encoding + Standard Scaling
- **Final Shape:** 16 features
- **Training Samples:** 400
- **Test Samples:** 100

### Approach 2: Label Encoding + Standard Scaling
- **Final Shape:** 13 features
- **Training Samples:** 400
- **Test Samples:** 100

## Feature Importance Analysis

### ANOVA F-test Results
- ExamResultPercent: 1758.9955
- Age: 13.8633
- Gender: 3.2077
- StudyHours: 2.0764
- EmploymentStatus: 1.6666
- Scholarship: 1.5667
- MockExamScore: 1.5597
- GPA: 1.4552
- ReviewCenter: 1.3551
- Confidence: 1.1838

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
