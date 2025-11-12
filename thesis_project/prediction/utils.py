import sys
import os
import numpy as np
import joblib

parent_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..',
    '..'
))

if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

# Import from models package
from models.regression_preprocessor import RegressionPreprocessor

def load_selected_model(model_name='random_forest', model_type='base'):
    try:
        if model_type == 'base':
            models_path = os.path.join(parent_path, 'models')
            model_dir = os.path.join(models_path, 'saved_base_models')
            model_file = f'{model_name}_model.pkl'
        else:
            models_path = os.path.join(parent_path, 'models')
            model_dir = os.path.join(models_path, 'saved_ensemble_models')
            model_file = f'{model_name}_ensemble.pkl'
        
        model_path = os.path.join(model_dir, model_file)
        preprocessor_path = os.path.join(
            models_path, 'saved_base_models', 'preprocessor.pkl'
        )
        
        if not os.path.exists(model_path):
            return None, None, f"Model file not found: {model_path}"
        
        if not os.path.exists(preprocessor_path):
            return None, None, f"Preprocessor file not found: {preprocessor_path}"
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        print(f"Loaded {model_type} model: {model_name}")
        return model, preprocessor, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Error loading model: {str(e)}"

def prepare_input_data(form_data):
    return {
        'Age': int(form_data['age']),
        'Gender': form_data['gender'],
        'StudyHours': int(form_data['study_hours']),
        'SleepHours': int(form_data['sleep_hours']),
        'ReviewCenter': int(form_data['review_center']),
        'Confidence': int(form_data['confidence']),
        'TestAnxiety': int(form_data['test_anxiety']),
        'MockExamScore': float(form_data['mock_exam_score']) if form_data.get('mock_exam_score') else None,
        'GPA': float(form_data['gpa']),
        'Scholarship': int(form_data['scholarship']),
        'InternshipGrade': float(form_data['internship_grade']),
        'IncomeLevel': form_data['income_level'],
        'EmploymentStatus': form_data['employment_status']
    }

def generate_recommendations(input_data, probability):
    recommendations = []
    
    if input_data['StudyHours'] < 5:
        recommendations.append({
            'type': 'critical',
            'title': 'Increase Study Hours',
            'message': 'Your daily study hours are below recommended levels. Aim for at least 5-6 hours.',
            'icon': 'fa-book'
        })
    
    if input_data['SleepHours'] < 6:
        recommendations.append({
            'type': 'warning',
            'title': 'Improve Sleep Quality',
            'message': 'Adequate sleep (7-8 hours) is crucial for memory retention and exam performance.',
            'icon': 'fa-bed'
        })
    
    if input_data['ReviewCenter'] == 0:
        recommendations.append({
            'type': 'info',
            'title': 'Consider Review Center',
            'message': 'Review centers provide structured learning and exam strategies.',
            'icon': 'fa-chalkboard-teacher'
        })
    
    if input_data['Confidence'] < 5:
        recommendations.append({
            'type': 'warning',
            'title': 'Build Confidence',
            'message': 'Practice more mock exams and focus on weak areas to boost confidence.',
            'icon': 'fa-chart-line'
        })
    
    if input_data['TestAnxiety'] > 7:
        recommendations.append({
            'type': 'critical',
            'title': 'Manage Test Anxiety',
            'message': 'High anxiety can affect performance. Consider stress management techniques and counseling.',
            'icon': 'fa-heart-pulse'
        })
    
    if input_data.get('MockExamScore') and input_data['MockExamScore'] < 70:
        recommendations.append({
            'type': 'critical',
            'title': 'Improve Mock Exam Scores',
            'message': 'Your mock exam scores indicate areas needing improvement. Focus on weak topics.',
            'icon': 'fa-clipboard-check'
        })
    
    if input_data['GPA'] < 2.5:
        recommendations.append({
            'type': 'warning',
            'title': 'Academic Foundation',
            'message': 'Review fundamental concepts from your coursework to strengthen your foundation.',
            'icon': 'fa-graduation-cap'
        })
    
    return recommendations

def get_risk_level(probability):
    if probability >= 0.75:
        return {
            'level': 'Low Risk',
            'color': 'green',
            'icon': 'fa-circle-check',
            'description': 'High likelihood of passing'
        }
    elif probability >= 0.50:
        return {
            'level': 'Moderate Risk',
            'color': 'yellow',
            'icon': 'fa-circle-exclamation',
            'description': 'Moderate likelihood of passing'
        }
    else:
        return {
            'level': 'High Risk',
            'color': 'red',
            'icon': 'fa-circle-xmark',
            'description': 'Low likelihood of passing'
        }

