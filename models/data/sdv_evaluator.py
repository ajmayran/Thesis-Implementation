import pandas as pd
import numpy as np
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.metadata import SingleTableMetadata
import warnings
warnings.filterwarnings('ignore')

class SDVEvaluator:
    
    def __init__(self, original_data_path, synthetic_data_path):
        self.original_data_path = original_data_path
        self.synthetic_data_path = synthetic_data_path
        self.original_data = None
        self.synthetic_data = None
        self.metadata = None
        
    def load_datasets(self):
        self.original_data = pd.read_csv(self.original_data_path)
        self.synthetic_data = pd.read_csv(self.synthetic_data_path)
        
        print(f"Original data shape: {self.original_data.shape}")
        print(f"Synthetic data shape: {self.synthetic_data.shape}")
        
        return self.original_data, self.synthetic_data
    
    def create_metadata(self):
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.original_data)
        
        self.metadata.update_column('Age', sdtype='numerical')
        self.metadata.update_column('Gender', sdtype='categorical')
        self.metadata.update_column('StudyHours', sdtype='numerical')
        self.metadata.update_column('SleepHours', sdtype='numerical')
        self.metadata.update_column('ReviewCenter', sdtype='categorical')
        self.metadata.update_column('MockExamScore', sdtype='numerical')
        self.metadata.update_column('GPA', sdtype='numerical')
        self.metadata.update_column('Scholarship', sdtype='categorical')
        self.metadata.update_column('InternshipGrade', sdtype='numerical')
        self.metadata.update_column('IncomeLevel', sdtype='categorical')
        self.metadata.update_column('EmploymentStatus', sdtype='categorical')
        self.metadata.update_column('Confidence', sdtype='numerical')
        self.metadata.update_column('TestAnxiety', sdtype='numerical')
        self.metadata.update_column('EnglishProficiency', sdtype='numerical')
        self.metadata.update_column('MotivationScore', sdtype='numerical')
        self.metadata.update_column('SocialSupport', sdtype='numerical')
        self.metadata.update_column('ExamResultPercent', sdtype='numerical')
        self.metadata.update_column('Passed', sdtype='categorical')
        
        return self.metadata
    
    def run_diagnostics(self):
        print("\n" + "="*60)
        print("RUNNING DIAGNOSTIC TESTS")
        print("="*60)
        
        diagnostic_report = run_diagnostic(
            real_data=self.original_data,
            synthetic_data=self.synthetic_data,
            metadata=self.metadata
        )
        
        print("\nDiagnostic Report:")
        print(diagnostic_report)
        
        return diagnostic_report
    
    def evaluate_data_quality(self):
        print("\n" + "="*60)
        print("EVALUATING DATA QUALITY")
        print("="*60)
        
        quality_report = evaluate_quality(
            real_data=self.original_data,
            synthetic_data=self.synthetic_data,
            metadata=self.metadata
        )
        
        print("\nQuality Report:")
        print(quality_report)
        
        return quality_report
    
    def statistical_comparison(self):
        print("\n" + "="*60)
        print("STATISTICAL COMPARISON")
        print("="*60)
        
        numerical_cols = ['Age', 'StudyHours', 'SleepHours', 'MockExamScore', 'GPA', 
                         'InternshipGrade', 'Confidence', 'TestAnxiety', 'EnglishProficiency',
                         'MotivationScore', 'SocialSupport', 'ExamResultPercent']
        
        for col in numerical_cols:
            if col in self.original_data.columns and col in self.synthetic_data.columns:
                print(f"\n{col}:")
                print(f"  Original - Mean: {self.original_data[col].mean():.2f}, Std: {self.original_data[col].std():.2f}")
                print(f"  Synthetic - Mean: {self.synthetic_data[col].mean():.2f}, Std: {self.synthetic_data[col].std():.2f}")
        
        categorical_cols = ['Gender', 'ReviewCenter', 'Scholarship', 'IncomeLevel', 
                           'EmploymentStatus', 'Passed']
        
        for col in categorical_cols:
            if col in self.original_data.columns and col in self.synthetic_data.columns:
                print(f"\n{col} Distribution:")
                print("Original:")
                print(self.original_data[col].value_counts(normalize=True).sort_index())
                print("Synthetic:")
                print(self.synthetic_data[col].value_counts(normalize=True).sort_index())
    
    def validate_pass_fail_rule(self):
        print("\n" + "="*60)
        print("VALIDATING PASS/FAIL RULE")
        print("="*60)
        
        violations = self.synthetic_data[
            ((self.synthetic_data['ExamResultPercent'] >= 70) & (self.synthetic_data['Passed'] == 0)) |
            ((self.synthetic_data['ExamResultPercent'] < 70) & (self.synthetic_data['Passed'] == 1))
        ]
        
        print(f"Total synthetic records: {len(self.synthetic_data)}")
        print(f"Rule violations: {len(violations)}")
        print(f"Compliance rate: {((len(self.synthetic_data) - len(violations)) / len(self.synthetic_data) * 100):.2f}%")
        
        if len(violations) > 0:
            print("\nViolation examples:")
            print(violations[['ExamResultPercent', 'Passed']].head())
        
        return violations

def evaluate_sdv_data():
    original_path = 'social_work_exam_dataset.csv'
    synthetic_path = 'sdv_dataset.csv'
    
    evaluator = SDVEvaluator(original_path, synthetic_path)
    
    evaluator.load_datasets()
    evaluator.create_metadata()
    
    diagnostic_report = evaluator.run_diagnostics()
    quality_report = evaluator.evaluate_data_quality()
    
    evaluator.statistical_comparison()
    evaluator.validate_pass_fail_rule()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    evaluate_sdv_data()