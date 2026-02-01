import pandas as pd
import numpy as np
import os

class SyntheticCVGenerator:
    
    def __init__(self, n_samples=140, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_cv_dataset(self):
        np.random.seed(self.random_state)
        
        n_pass = 500
        n_fail = 500
        
        age_dist = np.random.choice([21, 22, 23, 24, 25, 26], 
                                    self.n_samples, 
                                    p=[0.40, 0.34, 0.13, 0.10, 0.02, 0.01])
        
        gender_dist = np.random.choice(['Male', 'Female'], 
                                       self.n_samples, 
                                       p=[0.48, 0.52])
        
        study_hours = np.random.randint(1, 16, self.n_samples)
        sleep_hours = np.random.randint(4, 10, self.n_samples)
        
        review_center = np.random.choice([0, 1], 
                                        self.n_samples, 
                                        p=[0.45, 0.55])
        
        mock_exam_score = np.random.uniform(50, 89.99, self.n_samples)
        mock_exam_score = np.where(review_center == 0, np.nan, mock_exam_score)
        
        gpa_values = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.0]
        gpa = np.random.choice(gpa_values, self.n_samples)
        
        scholarship = np.random.choice([0, 1], 
                                      self.n_samples, 
                                      p=[0.55, 0.45])
        
        internship_values = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.0]
        internship_grade = np.random.choice(internship_values, self.n_samples)
        
        income_level = np.random.choice(['Low', 'Middle', 'High'], 
                                       self.n_samples, 
                                       p=[0.35, 0.45, 0.20])
        
        employment_status = np.random.choice(['Unemployed', 'Unskilled', 'Skilled', 'Professional'], 
                                            self.n_samples, 
                                            p=[0.25, 0.30, 0.30, 0.15])
        
        confidence = np.random.randint(1, 11, self.n_samples)
        test_anxiety = np.random.randint(1, 11, self.n_samples)
        english_proficiency = np.random.randint(1, 11, self.n_samples)
        motivation_score = np.random.randint(1, 11, self.n_samples)
        family_support = np.random.randint(1, 11, self.n_samples)
        
        exam_result_pass = np.random.uniform(75.00, 89.80, n_pass)
        exam_result_fail = np.random.uniform(40.00, 74.99, n_fail)
        
        exam_result_percent = np.concatenate([exam_result_pass, exam_result_fail])
        passed = np.concatenate([np.ones(n_pass, dtype=int), np.zeros(n_fail, dtype=int)])
        
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)
        
        exam_result_percent = exam_result_percent[indices]
        passed = passed[indices]
        
        exam_result_percent = np.round(exam_result_percent, 2)
        
        df = pd.DataFrame({
            'Age': age_dist,
            'Gender': gender_dist,
            'StudyHours': study_hours,
            'SleepHours': sleep_hours,
            'ReviewCenter': review_center,
            'MockExamScore': np.round(mock_exam_score, 2),
            'GPA': gpa,
            'Scholarship': scholarship,
            'InternshipGrade': internship_grade,
            'IncomeLevel': income_level,
            'EmploymentStatus': employment_status,
            'Confidence': confidence,
            'TestAnxiety': test_anxiety,
            'EnglishProficiency': english_proficiency,
            'MotivationScore': motivation_score,
            'FamilySupport': family_support,
            'ExamResultPercent': exam_result_percent,
            'Passed': passed
        })
        
        return df
    
    def save_data(self, df, filename='synthetic_cv_dataset.csv', output_dir='data'):
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"Synthetic CV dataset saved to {filepath}")
        print(f"Shape: {df.shape}")
        print(f"\nFirst few rows:\n{df.head(10)}")
        print(f"\nDataset statistics:\n{df.describe()}")
        print(f"\nPass/Fail distribution:")
        print(df['Passed'].value_counts())
        print(f"\nPercentage passed: {(df['Passed'].sum() / len(df) * 100):.2f}%")
        print(f"\nExam score range: {df['ExamResultPercent'].min():.2f} - {df['ExamResultPercent'].max():.2f}")
        
        return filepath

def generate_cv_dataset():
    generator = SyntheticCVGenerator(n_samples=140, random_state=42)
    
    print("Generating CV dataset with 500 pass and 500 fail...")
    
    cv_data = generator.generate_cv_dataset()
    
    filepath = generator.save_data(cv_data, filename='social_work_exam_dataset.csv')
    
    print(f"\nDataset generated successfully at: {filepath}")
    print(f"Total samples: {len(cv_data)}")
    print(f"Total features: {len(cv_data.columns) - 2}")
    print(f"Target columns: ExamResultPercent, Passed")

if __name__ == "__main__":
    generate_cv_dataset()