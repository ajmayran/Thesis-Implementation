import sys
import os
import numpy as np
import joblib
import pandas as pd

parent_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..',
    '..'
))

if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

def load_selected_model(model_name='random_forest', model_type='classification_base'):
    try:
        models_path = os.path.join(parent_path, 'models')
        
        if model_type == 'classification_base':
            model_dir = os.path.join(models_path, 'saved_models')
            model_file = f'{model_name}_model.pkl'
            preprocessor_dir = model_dir
            preprocessor_file = 'preprocessing_objects.pkl'
            
        elif model_type == 'classification_ensemble':
            model_dir = os.path.join(models_path, 'saved_classification_ensemble_models')
            model_file = f'{model_name}_ensemble.pkl'
            preprocessor_dir = os.path.join(models_path, 'classification_processed_data')
            preprocessor_file = 'preprocessing_objects.pkl'
        else:
            return None, None, f"Invalid model_type: {model_type}. Use 'classification_base' or 'classification_ensemble'"
        
        model_path = os.path.join(model_dir, model_file)
        preprocessor_path = os.path.join(preprocessor_dir, preprocessor_file)
        
        if not os.path.exists(model_path):
            return None, None, f"Model file not found: {model_path}"
        
        if not os.path.exists(preprocessor_path):
            return None, None, f"Preprocessor file not found: {preprocessor_path}"
        
        model = joblib.load(model_path)
        preprocessing_objects = joblib.load(preprocessor_path)
        
        print(f"[LOAD] Model: {model_file} from {model_dir}")
        print(f"[LOAD] Preprocessor from: {preprocessor_path}")
        
        return model, preprocessing_objects, None
        
    except Exception as e:
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
            'message': f'Your predicted pass probability is {pass_probability*100:.1f}%. Maintain your current efforts.'
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
    
    if input_data.get('TestAnxiety', 5) > 7:
        recommendations.append({
            'icon': 'fa-heart',
            'color': 'red',
            'title': 'Manage Test Anxiety',
            'message': 'Consider stress management techniques, breathing exercises, or counseling to reduce anxiety.'
        })
    
    if input_data.get('MockExamScore', 0) and input_data.get('MockExamScore') < 70:
        recommendations.append({
            'icon': 'fa-chart-line',
            'color': 'orange',
            'title': 'Improve Mock Exam Performance',
            'message': 'Focus on weak areas identified in mock exams. Practice more sample questions.'
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