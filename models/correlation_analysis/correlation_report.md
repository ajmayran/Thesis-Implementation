# Correlation Analysis Report

## Overview

Target Variable: **Passed** (Binary)

## Statistical Methods Applied

1. **Pearson Correlation (r)**: Continuous → Continuous
2. **Point-Biserial Correlation (rpb)**: Continuous → Binary Target
3. **Chi-Square Test + Cramer's V**: Categorical → Binary Target
4. **Cramer's V**: Categorical → Categorical
5. **One-Way ANOVA**: Categorical → Continuous

## Point-Biserial Correlation (Continuous → Binary Target)

| Feature | Correlation (rpb) | P-Value | Significant |
|---------|-------------------|---------|-------------|
| MockExamScore | 0.1820 | 0.0013 | Yes |
| GPA | 0.1108 | 0.0096 | Yes |
| StudyHours | 0.0824 | 0.0543 | No |
| TestAnxiety | 0.0724 | 0.0910 | No |
| InternshipGrade | 0.0481 | 0.2617 | No |
| Scholarship | 0.0472 | 0.2705 | No |
| Confidence | 0.0352 | 0.4113 | No |
| Age | 0.0291 | 0.4976 | No |
| ReviewCenter | 0.0178 | 0.6784 | No |
| SleepHours | 0.0082 | 0.8487 | No |

## Chi-Square Test + Cramer's V (Categorical → Binary Target)

| Feature | Chi-Square | P-Value | Cramer's V | Significant |
|---------|------------|---------|------------|-------------|
| EmploymentStatus | 5.6618 | 0.1293 | 0.1018 | No |
| IncomeLevel | 3.6282 | 0.1630 | 0.0815 | No |
| Gender | 2.6428 | 0.1040 | 0.0696 | No |

## Cramer's V (Categorical → Categorical)

| Feature 1 | Feature 2 | Cramer's V | P-Value | Significant |
|-----------|-----------|------------|---------|-------------|
| IncomeLevel | EmploymentStatus | 0.1539 | 0.0002 | Yes |
| Gender | IncomeLevel | 0.0638 | 0.3286 | No |
| Gender | EmploymentStatus | 0.0462 | 0.7608 | No |

## One-Way ANOVA (Categorical → Continuous)

| Categorical Feature | Numerical Feature | F-Statistic | P-Value | Significant |
|---------------------|-------------------|-------------|---------|-------------|
| EmploymentStatus | Age | 6.0055 | 0.0005 | Yes |
| IncomeLevel | StudyHours | 5.6406 | 0.0038 | Yes |
| EmploymentStatus | GPA | 5.1788 | 0.0016 | Yes |
| EmploymentStatus | StudyHours | 5.1767 | 0.0016 | Yes |
| IncomeLevel | GPA | 4.5230 | 0.0113 | Yes |
| IncomeLevel | SleepHours | 4.0963 | 0.0172 | Yes |
| IncomeLevel | Age | 2.8870 | 0.0566 | No |
| EmploymentStatus | MockExamScore | 1.7895 | 0.1491 | No |
| EmploymentStatus | SleepHours | 1.5330 | 0.2049 | No |
| EmploymentStatus | InternshipGrade | 1.4659 | 0.2229 | No |
| IncomeLevel | MockExamScore | 1.4602 | 0.2338 | No |
| IncomeLevel | TestAnxiety | 1.3623 | 0.2570 | No |
| IncomeLevel | InternshipGrade | 1.2871 | 0.2769 | No |
| EmploymentStatus | Confidence | 0.9700 | 0.4065 | No |
| Gender | MockExamScore | 0.7419 | 0.3897 | No |
| EmploymentStatus | TestAnxiety | 0.6779 | 0.5658 | No |
| Gender | Confidence | 0.5517 | 0.4579 | No |
| Gender | SleepHours | 0.4953 | 0.4819 | No |
| Gender | StudyHours | 0.2123 | 0.6451 | No |
| Gender | TestAnxiety | 0.1089 | 0.7416 | No |

## Inter-Feature Correlations

No high inter-feature correlations (>0.5) detected.

## Files Generated

- `correlation_results.json` - Complete correlation data
- `correlation_matrix.csv` - Pearson correlation matrix (continuous variables)
- `correlation_matrix.png` - Heatmap visualization
- `target_correlations.png` - Point-biserial correlations plot
- `categorical_associations.png` - Cramer's V associations plot
- `anova_results.png` - One-Way ANOVA results plot
- `correlation_report.md` - This report
