# Correlation Analysis Report

## Overview

Target Variable: **Passed** (Binary)

### Feature Groups

- **Numerical Features**: Age, StudyHours, SleepHours, Confidence, MockExamScore, GPA, InternshipGrade, TestAnxiety, SocialSupport, MotivationScore, EnglishProficiency, ExamResultPercent
- **Binary Features**: ReviewCenter, Scholarship
- **Categorical Features**: Gender, IncomeLevel, EmploymentStatus

## Statistical Methods Applied

1. **Pearson Correlation (r)**: Continuous -> Continuous
2. **Point-Biserial Correlation (rpb)**: Continuous -> Binary Target
3. **Chi-Square Test + Cramer's V**: Categorical -> Binary Target
4. **Cramer's V**: Categorical -> Categorical
5. **One-Way ANOVA**: Categorical -> Continuous

## Point-Biserial Correlation (Continuous -> Binary Target)

| Feature | Correlation (rpb) | P-Value | Significant |
|---------|-------------------|---------|-------------|
| ExamResultPercent | 0.3039 | 0.0000 | Yes |
| Age | -0.1113 | 0.0002 | Yes |
| ReviewCenter | 0.1051 | 0.0005 | Yes |
| TestAnxiety | 0.0827 | 0.0059 | Yes |
| MockExamScore | 0.0692 | 0.0862 | No |
| SleepHours | -0.0630 | 0.0360 | Yes |
| InternshipGrade | 0.0603 | 0.0475 | Yes |
| GPA | -0.0502 | 0.0945 | No |
| StudyHours | 0.0485 | 0.1065 | No |
| EnglishProficiency | 0.0403 | 0.1801 | No |
| MotivationScore | 0.0213 | 0.4794 | No |
| Confidence | -0.0108 | 0.7202 | No |
| SocialSupport | 0.0103 | 0.7330 | No |
| Scholarship | -0.0044 | 0.8840 | No |

## Chi-Square Test + Cramer's V (Categorical -> Binary Target)

| Feature | Chi-Square | P-Value | Cramer's V | Significant |
|---------|------------|---------|------------|-------------|
| EmploymentStatus | 10.2633 | 0.0059 | 0.0962 | Yes |
| IncomeLevel | 1.2424 | 0.5373 | 0.0335 | No |
| Gender | 0.0599 | 0.8066 | 0.0073 | No |

## Cramer's V (Categorical -> Categorical)

| Feature 1 | Feature 2 | Cramer's V | P-Value | Significant |
|-----------|-----------|------------|---------|-------------|
| Gender | IncomeLevel | 0.0892 | 0.0121 | Yes |
| Gender | EmploymentStatus | 0.0295 | 0.6177 | No |
| IncomeLevel | EmploymentStatus | 0.0188 | 0.9403 | No |

## One-Way ANOVA (Categorical -> Continuous)

| Categorical Feature | Numerical Feature | F-Statistic | P-Value | Significant |
|---------------------|-------------------|-------------|---------|-------------|
| Gender | StudyHours | 12.8272 | 0.0004 | Yes |
| Gender | SocialSupport | 11.6406 | 0.0007 | Yes |
| IncomeLevel | Confidence | 11.6368 | 0.0000 | Yes |
| EmploymentStatus | StudyHours | 11.6104 | 0.0000 | Yes |
| IncomeLevel | MockExamScore | 11.4449 | 0.0000 | Yes |
| EmploymentStatus | Age | 7.8610 | 0.0004 | Yes |
| Gender | MotivationScore | 6.5602 | 0.0106 | Yes |
| IncomeLevel | EnglishProficiency | 5.8222 | 0.0031 | Yes |
| IncomeLevel | SocialSupport | 4.3524 | 0.0131 | Yes |
| Gender | InternshipGrade | 4.3124 | 0.0381 | Yes |
| Gender | GPA | 4.0412 | 0.0446 | Yes |
| EmploymentStatus | ExamResultPercent | 3.4590 | 0.0318 | Yes |
| EmploymentStatus | MotivationScore | 3.3640 | 0.0350 | Yes |
| IncomeLevel | MotivationScore | 3.1331 | 0.0440 | Yes |
| EmploymentStatus | InternshipGrade | 2.7824 | 0.0623 | No |
| EmploymentStatus | Confidence | 2.2481 | 0.1061 | No |
| IncomeLevel | TestAnxiety | 2.0998 | 0.1230 | No |
| IncomeLevel | SleepHours | 1.9143 | 0.1479 | No |
| Gender | SleepHours | 1.6327 | 0.2016 | No |
| IncomeLevel | GPA | 1.4363 | 0.2383 | No |

## ExamResultPercent Correlations (Top Features)

| Feature | Pearson r | P-Value |
|---------|-----------|----------|
| Age | -0.2068 | 0.0000 |
| TestAnxiety | 0.1814 | 0.0000 |
| StudyHours | 0.1812 | 0.0000 |
| EnglishProficiency | 0.1586 | 0.0000 |
| SleepHours | -0.1583 | 0.0000 |
| GPA | -0.1340 | 0.0000 |

## Inter-Feature Correlations

No high inter-feature correlations (>0.5) detected.

## Files Generated

- `correlation_results.json` - Complete correlation data
- `correlation_matrix.csv` - Pearson correlation matrix (continuous + binary variables)
- `correlation_matrix.png` - Heatmap visualization
- `target_correlations.png` - Point-biserial correlations plot
- `categorical_associations.png` - Cramer's V associations plot
- `anova_results.png` - One-Way ANOVA results plot
- `exam_result_correlations.png` - ExamResultPercent scatter plots
- `correlation_report.md` - This report
