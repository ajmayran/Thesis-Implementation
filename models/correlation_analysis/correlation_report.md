# Correlation Analysis Report

## Overview

**Target Variable**: `ExamResultPercent` (Continuous - Exam Score Percentage)

**Secondary Variable**: `Passed` (Binary - Pass/Fail, threshold: 70%)

### Feature Groups

- **Numerical Features** (11): Age, StudyHours, SleepHours, Confidence, MockExamScore, GPA, InternshipGrade, TestAnxiety, SocialSupport, MotivationScore, EnglishProficiency
- **Binary Features** (2): ReviewCenter, Scholarship
- **Categorical Features** (3): Gender, IncomeLevel, EmploymentStatus

## Statistical Methods Applied

| Method | Variable Types | Purpose |
|--------|---------------|----------|
| Pearson Correlation (r) | Continuous/Binary -> Continuous Target | Measures linear relationship strength |
| Spearman Rank Correlation (rho) | Continuous/Binary -> Continuous Target | Captures monotonic (including non-linear) relationships |
| Point-Biserial Correlation (rpb) | Continuous -> Binary (Passed) | Supplementary analysis for pass/fail outcome |
| Chi-Square + Cramer's V | Categorical -> Binary (Passed) | Tests categorical feature association with pass/fail |
| One-Way ANOVA + Eta-squared | Categorical -> Continuous Target | Tests mean score differences across categorical groups |
| Kruskal-Wallis H | Categorical -> Continuous Target | Non-parametric alternative to ANOVA |
| Cramer's V (Inter-feature) | Categorical -> Categorical | Checks categorical feature independence |

## Pearson Correlation with ExamResultPercent

| Rank | Feature | Pearson r | P-Value | Significant | Strength |
|------|---------|-----------|---------|-------------|----------|
| 1 | Age | -0.2068 | 0.000000 | Yes | Weak |
| 2 | TestAnxiety | 0.1814 | 0.000000 | Yes | Weak |
| 3 | StudyHours | 0.1812 | 0.000000 | Yes | Weak |
| 4 | ReviewCenter | 0.1662 | 0.000000 | Yes | Weak |
| 5 | EnglishProficiency | 0.1586 | 0.000000 | Yes | Weak |
| 6 | SleepHours | -0.1583 | 0.000000 | Yes | Weak |
| 7 | GPA | -0.1340 | 0.000007 | Yes | Weak |
| 8 | MockExamScore | -0.0572 | 0.156147 | No | Negligible |
| 9 | SocialSupport | 0.0551 | 0.066372 | No | Negligible |
| 10 | Confidence | -0.0518 | 0.084408 | No | Negligible |
| 11 | InternshipGrade | 0.0198 | 0.515515 | No | Negligible |
| 12 | Scholarship | -0.0148 | 0.622551 | No | Negligible |
| 13 | MotivationScore | 0.0027 | 0.928263 | No | Negligible |

## Spearman Rank Correlation with ExamResultPercent

| Rank | Feature | Spearman rho | P-Value | Significant | Strength |
|------|---------|--------------|---------|-------------|----------|
| 1 | StudyHours | 0.1829 | 0.000000 | Yes | Weak |
| 2 | Age | -0.1828 | 0.000000 | Yes | Weak |
| 3 | TestAnxiety | 0.1620 | 0.000000 | Yes | Weak |
| 4 | EnglishProficiency | 0.1554 | 0.000000 | Yes | Weak |
| 5 | ReviewCenter | 0.1464 | 0.000001 | Yes | Weak |
| 6 | SleepHours | -0.1401 | 0.000003 | Yes | Weak |
| 7 | GPA | -0.1226 | 0.000042 | Yes | Weak |
| 8 | MockExamScore | -0.0725 | 0.072293 | No | Negligible |
| 9 | Confidence | -0.0526 | 0.079935 | No | Negligible |
| 10 | SocialSupport | 0.0511 | 0.088751 | No | Negligible |
| 11 | Scholarship | -0.0135 | 0.654098 | No | Negligible |
| 12 | InternshipGrade | 0.0100 | 0.743239 | No | Negligible |
| 13 | MotivationScore | 0.0052 | 0.862108 | No | Negligible |

## Pearson vs Spearman Comparison (Non-Linearity Detection)

| Feature | Pearson r | Spearman rho | Abs Difference | Non-Linear? |
|---------|-----------|--------------|----------------|-------------|
| Age | -0.2068 | -0.1828 | 0.0240 | No |
| ReviewCenter | 0.1662 | 0.1464 | 0.0198 | No |
| TestAnxiety | 0.1814 | 0.1620 | 0.0195 | No |
| SleepHours | -0.1583 | -0.1401 | 0.0182 | No |
| MockExamScore | -0.0572 | -0.0725 | 0.0153 | No |
| GPA | -0.1340 | -0.1226 | 0.0114 | No |
| InternshipGrade | 0.0198 | 0.0100 | 0.0098 | No |
| SocialSupport | 0.0551 | 0.0511 | 0.0040 | No |
| EnglishProficiency | 0.1586 | 0.1554 | 0.0032 | No |
| MotivationScore | 0.0027 | 0.0052 | 0.0025 | No |
| StudyHours | 0.1812 | 0.1829 | 0.0017 | No |
| Scholarship | -0.0148 | -0.0135 | 0.0013 | No |
| Confidence | -0.0518 | -0.0526 | 0.0008 | No |

## Point-Biserial Correlation (Features -> Passed) [Supplementary]

| Feature | Correlation (rpb) | P-Value | Significant |
|---------|-------------------|---------|-------------|
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

## Chi-Square Test + Cramer's V (Categorical -> Passed)

| Feature | Chi-Square | P-Value | Cramer's V | Significant |
|---------|------------|---------|------------|-------------|
| EmploymentStatus | 10.2633 | 0.0059 | 0.0962 | Yes |
| IncomeLevel | 1.2424 | 0.5373 | 0.0335 | No |
| Gender | 0.0599 | 0.8066 | 0.0073 | No |

## One-Way ANOVA (Categorical -> Continuous)

### ANOVA for ExamResultPercent

| Categorical Feature | F-Statistic | P-Value | Eta-Squared | Effect Size | Significant |
|---------------------|-------------|---------|-------------|-------------|-------------|
| EmploymentStatus | 3.4590 | 0.0318 | 0.0062 | Negligible | Yes |
| IncomeLevel | 0.4591 | 0.6319 | 0.0008 | Negligible | No |
| Gender | 0.1081 | 0.7423 | 0.0001 | Negligible | No |

### All ANOVA Results (Top 20 by F-Statistic)

| Categorical Feature | Numerical Feature | F-Statistic | P-Value | Eta-Squared | Significant |
|---------------------|-------------------|-------------|---------|-------------|-------------|
| Gender | StudyHours | 12.8272 | 0.0004 | 0.0114 | Yes |
| Gender | SocialSupport | 11.6406 | 0.0007 | 0.0104 | Yes |
| IncomeLevel | Confidence | 11.6368 | 0.0000 | 0.0206 | Yes |
| EmploymentStatus | StudyHours | 11.6104 | 0.0000 | 0.0205 | Yes |
| IncomeLevel | MockExamScore | 11.4449 | 0.0000 | 0.0360 | Yes |
| EmploymentStatus | Age | 7.8610 | 0.0004 | 0.0140 | Yes |
| Gender | MotivationScore | 6.5602 | 0.0106 | 0.0059 | Yes |
| IncomeLevel | EnglishProficiency | 5.8222 | 0.0031 | 0.0104 | Yes |
| IncomeLevel | SocialSupport | 4.3524 | 0.0131 | 0.0078 | Yes |
| Gender | InternshipGrade | 4.3124 | 0.0381 | 0.0040 | Yes |
| Gender | GPA | 4.0412 | 0.0446 | 0.0036 | Yes |
| EmploymentStatus | ExamResultPercent | 3.4590 | 0.0318 | 0.0062 | Yes |
| EmploymentStatus | MotivationScore | 3.3640 | 0.0350 | 0.0060 | Yes |
| IncomeLevel | MotivationScore | 3.1331 | 0.0440 | 0.0056 | Yes |
| EmploymentStatus | InternshipGrade | 2.7824 | 0.0623 | 0.0051 | No |
| EmploymentStatus | Confidence | 2.2481 | 0.1061 | 0.0040 | No |
| IncomeLevel | TestAnxiety | 2.0998 | 0.1230 | 0.0038 | No |
| IncomeLevel | SleepHours | 1.9143 | 0.1479 | 0.0034 | No |
| Gender | SleepHours | 1.6327 | 0.2016 | 0.0015 | No |
| IncomeLevel | GPA | 1.4363 | 0.2383 | 0.0026 | No |

## Kruskal-Wallis H Test (Categorical -> ExamResultPercent) [Non-Parametric]

| Feature | H-Statistic | P-Value | Significant |
|---------|-------------|---------|-------------|
| EmploymentStatus | 5.3542 | 0.0688 | No |
| IncomeLevel | 0.5279 | 0.7680 | No |
| Gender | 0.4399 | 0.5072 | No |

## Cramer's V (Categorical Inter-Feature Analysis)

| Feature 1 | Feature 2 | Cramer's V | P-Value | Significant |
|-----------|-----------|------------|---------|-------------|
| Gender | IncomeLevel | 0.0892 | 0.0121 | Yes |
| Gender | EmploymentStatus | 0.0295 | 0.6177 | No |
| IncomeLevel | EmploymentStatus | 0.0188 | 0.9403 | No |

## Inter-Feature Correlations (Multicollinearity Check)

No high inter-feature correlations (|r| > 0.5) detected among predictor variables. This confirms that multicollinearity is not a concern in the dataset.

## ExamResultPercent Top Correlated Features (Summary)

| Feature | Pearson r | Pearson p | Spearman rho | Spearman p |
|---------|-----------|-----------|--------------|------------|
| Age | -0.2068 | 0.000000 | -0.1828 | 0.000000 |
| TestAnxiety | 0.1814 | 0.000000 | 0.1620 | 0.000000 |
| StudyHours | 0.1812 | 0.000000 | 0.1829 | 0.000000 |
| ReviewCenter | 0.1662 | 0.000000 | 0.1464 | 0.000001 |
| EnglishProficiency | 0.1586 | 0.000000 | 0.1554 | 0.000000 |
| SleepHours | -0.1583 | 0.000000 | -0.1401 | 0.000003 |

## Files Generated

| File | Description |
|------|-------------|
| `correlation_results.json` | Complete correlation data (all methods) |
| `correlation_matrix.csv` | Pearson correlation matrix |
| `correlation_matrix.png` | Correlation matrix heatmap |
| `target_correlations_pearson.png` | Pearson correlations with target bar chart |
| `pearson_vs_spearman.png` | Pearson vs Spearman comparison (3-panel) |
| `categorical_associations.png` | Cramer's V associations bar chart |
| `anova_results.png` | One-Way ANOVA results bar chart |
| `anova_target_boxplots.png` | Box plots of target by categorical features |
| `exam_result_correlations.png` | Scatter plots of target vs top features |
| `correlation_report.md` | This report |
