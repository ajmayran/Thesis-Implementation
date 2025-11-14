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

def generate_recommendations(input_data, pass_probability):
    recommendations = []
    
    if pass_probability < 0.5:
        recommendations.append({
            'icon': 'fa-exclamation-triangle',
            'color': 'red',
            'title': 'High Risk - Immediate Action Required',
            'message': 'Your predicted pass probability is below 50%. Significant improvements needed in multiple areas.'
        })
    elif pass_probability < 0.75:
        recommendations.append({
            'icon': 'fa-triangle-exclamation',
            'color': 'orange',
            'title': 'Moderate Risk - Improvement Needed',
            'message': 'Your pass probability is moderate. Focus on key improvement areas to increase your chances.'
        })
    else:
        recommendations.append({
            'icon': 'fa-check-circle',
            'color': 'green',
            'title': 'Good Standing - Keep It Up!',
            'message': 'Your predicted pass probability is 74.7%. Maintain your current efforts.'
        })
    
    if input_data.get('StudyHours', 0) < 8:
        recommendations.append({
            'icon': 'fa-book',
            'color': 'blue',
            'title': 'Increase Study Time',
            'message': 'Students with 8-10 hours of daily study have significantly higher pass rates.'
        })
    
    if input_data.get('SleepHours', 0) < 6:
        recommendations.append({
            'icon': 'fa-bed',
            'color': 'purple',
            'title': 'Improve Sleep Quality',
            'message': 'Getting 7-8 hours of sleep improves memory retention and exam performance.'
        })
    
    if input_data.get('ReviewCenter') == 0:
        recommendations.append({
            'icon': 'fa-school',
            'color': 'blue',
            'title': 'Consider Review Center',
            'message': 'Attending a review center increases pass likelihood by 25-30%.'
        })
    
    if input_data.get('Confidence', 3) < 3:
        recommendations.append({
            'icon': 'fa-brain',
            'color': 'orange',
            'title': 'Build Confidence',
            'message': 'Practice mock exams regularly to boost confidence and identify weak areas.'
        })
    
    return recommendations

def get_risk_level(probability):
    if probability >= 0.75:
        return {
            'level': 'Low Risk',
            'color': 'green',
            'message': 'High likelihood of passing. Continue with current preparation strategy.',
            'icon': 'fa-circle-check'
        }
    elif probability >= 0.5:
        return {
            'level': 'Moderate Risk',
            'color': 'orange',
            'message': 'Moderate pass likelihood. Focus on improvement areas identified below.',
            'icon': 'fa-triangle-exclamation'
        }
    else:
        return {
            'level': 'High Risk',
            'color': 'red',
            'message': 'Low pass likelihood. Immediate action required in multiple areas.',
            'icon': 'fa-circle-exclamation'
        }