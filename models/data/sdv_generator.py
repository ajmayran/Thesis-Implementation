import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import os

class SDVDataGenerator:
    
    def __init__(self, data_path, random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.original_data = None
        self.metadata = None
        
    def load_data(self):
        self.original_data = pd.read_csv(self.data_path)
        
        print(f"Loaded data shape: {self.original_data.shape}")
        print(f"Columns: {list(self.original_data.columns)}")
        print(f"\nData types:\n{self.original_data.dtypes}")
        print(f"\nMissing values:\n{self.original_data.isnull().sum()}")
        print(f"\nSample data:\n{self.original_data.head()}")
        
        return self.original_data
    
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
        self.metadata.update_column('FamilySupport', sdtype='numerical')
        self.metadata.update_column('ExamResultPercent', sdtype='numerical')
        self.metadata.update_column('Passed', sdtype='categorical')
        
        print("\nMetadata created successfully")
        return self.metadata
    
    def generate_with_gaussian_copula(self, n_samples=1000):
        print(f"\nGenerating {n_samples} samples using Gaussian Copula...")
        
        synthesizer = GaussianCopulaSynthesizer(
            metadata=self.metadata,
            default_distribution='norm'
        )
        
        synthesizer.fit(self.original_data)
        synthetic_data = synthesizer.sample(num_rows=n_samples)
        
        return synthetic_data
    
    def post_process(self, synthetic_data):
        synthetic_data['Age'] = synthetic_data['Age'].clip(21, 26).round().astype(int)
        synthetic_data['StudyHours'] = synthetic_data['StudyHours'].clip(1, 10).round().astype(int)
        synthetic_data['SleepHours'] = synthetic_data['SleepHours'].clip(1, 9).round().astype(int)
        
        if synthetic_data['ReviewCenter'].dtype in ['float64', 'int64']:
            synthetic_data['ReviewCenter'] = synthetic_data['ReviewCenter'].round().astype(int)
        
        if synthetic_data['Scholarship'].dtype in ['float64', 'int64']:
            synthetic_data['Scholarship'] = synthetic_data['Scholarship'].round().astype(int)
        
        synthetic_data.loc[synthetic_data['ReviewCenter'] == 0, 'MockExamScore'] = np.nan
        mask = (synthetic_data['ReviewCenter'] == 1) | (synthetic_data['ReviewCenter'] == '1')
        if mask.any():
            synthetic_data.loc[mask, 'MockExamScore'] = synthetic_data.loc[mask, 'MockExamScore'].clip(60, 100)
        
        gpa_values = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.0]
        synthetic_data['GPA'] = synthetic_data['GPA'].apply(
            lambda x: min(gpa_values, key=lambda gpa: abs(gpa - x))
        )
        
        internship_values = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.0]
        synthetic_data['InternshipGrade'] = synthetic_data['InternshipGrade'].apply(
            lambda x: min(internship_values, key=lambda grade: abs(grade - x))
        )
        
        synthetic_data['Confidence'] = synthetic_data['Confidence'].clip(1, 10).round().astype(int)
        synthetic_data['TestAnxiety'] = synthetic_data['TestAnxiety'].clip(1, 10).round().astype(int)
        synthetic_data['EnglishProficiency'] = synthetic_data['EnglishProficiency'].clip(1, 10).round().astype(int)
        synthetic_data['MotivationScore'] = synthetic_data['MotivationScore'].clip(1, 10).round().astype(int)
        synthetic_data['FamilySupport'] = synthetic_data['FamilySupport'].clip(1, 10).round().astype(int)
        
        synthetic_data['ExamResultPercent'] = synthetic_data['ExamResultPercent'].clip(60, 100).round(2)
        
        synthetic_data['Passed'] = (synthetic_data['ExamResultPercent'] >= 70).astype(int)
        
        return synthetic_data
    
    def save_data(self, synthetic_data, filename, output_dir=''):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename
            
        synthetic_data.to_csv(filepath, index=False)
        
        print(f"\nSynthetic data saved to {filepath}")
        print(f"Shape: {synthetic_data.shape}")
        print(f"\nFirst few rows:\n{synthetic_data.head()}")
        print(f"\nPass/Fail distribution:")
        print(synthetic_data['Passed'].value_counts())
        print(f"\nGender distribution:")
        print(synthetic_data['Gender'].value_counts())
        print(f"\nIncomeLevel distribution:")
        print(synthetic_data['IncomeLevel'].value_counts())
        print(f"\nEmploymentStatus distribution:")
        print(synthetic_data['EmploymentStatus'].value_counts())
        
        return filepath
    
    def compare_distributions(self, synthetic_data):
        print("\n" + "="*60)
        print("DISTRIBUTION COMPARISON")
        print("="*60)
        
        for col in self.original_data.columns:
            if col in ['Gender', 'IncomeLevel', 'EmploymentStatus', 'Passed', 'ReviewCenter', 'Scholarship']:
                print(f"\n{col}:")
                print("Original:")
                print(self.original_data[col].value_counts(normalize=True).sort_index())
                print("\nSynthetic:")
                print(synthetic_data[col].value_counts(normalize=True).sort_index())

def generate_sdv_datasets():
    data_path = 'social_work_exam_dataset.csv'
    generator = SDVDataGenerator(data_path, random_state=42)
    
    generator.load_data()
    generator.create_metadata()
    
    synthetic_copula = generator.generate_with_gaussian_copula(n_samples=1000)
    synthetic_copula = generator.post_process(synthetic_copula)
    
    generator.save_data(synthetic_copula, 'sdv_dataset.csv')
    generator.compare_distributions(synthetic_copula)
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Dataset saved: sdv_dataset.csv")

if __name__ == "__main__":
    generate_sdv_datasets()