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
    """
    Prepare input data for prediction including new fields
    """
    input_data = {
        'Age': form_data['age'],
        'Gender': form_data['gender'],
        'StudyHours': form_data['study_hours'],
        'SleepHours': form_data['sleep_hours'],
        'ReviewCenter': 1 if form_data['review_center'] else 0,
        'Confidence': form_data['confidence'],
        'TestAnxiety': form_data['test_anxiety'],
        'MockExamScore': form_data.get('mock_exam_score'),
        'GPA': form_data['gpa'],
        'Scholarship': 1 if form_data['scholarship'] else 0,
        'InternshipGrade': form_data['internship_grade'],
        'IncomeLevel': form_data['income_level'],
        'EmploymentStatus': form_data['employment_status'],
        'EnglishProficiency': form_data['english_proficiency'],
        'MotivationScore': form_data['motivation_score'],
        'SocialSupport': form_data['social_support']
    }
    
    return input_data

def generate_recommendations(input_data, pass_probability):
    """
    Generate personalized recommendations based on input data and prediction
    """
    recommendations = []
    
    # Study hours recommendation
    if input_data['StudyHours'] < 5:
        recommendations.append({
            'category': 'Study Habits',
            'priority': 'high',
            'message': 'Increase your daily study hours. Aim for at least 6-8 hours of focused study.',
            'action': 'Create a structured study schedule'
        })
    
    # Sleep hours recommendation
    if input_data['SleepHours'] < 6:
        recommendations.append({
            'category': 'Health & Wellness',
            'priority': 'high',
            'message': 'Insufficient sleep can affect cognitive performance. Aim for 7-8 hours of sleep.',
            'action': 'Establish a regular sleep schedule'
        })
    
    # Mock exam recommendation
    if input_data.get('MockExamScore') and input_data['MockExamScore'] < 70:
        recommendations.append({
            'category': 'Test Performance',
            'priority': 'high',
            'message': 'Your mock exam scores indicate areas for improvement. Focus on weak topics.',
            'action': 'Take more practice tests and review incorrect answers'
        })
    
    # Review center recommendation
    if not input_data['ReviewCenter'] and pass_probability < 0.7:
        recommendations.append({
            'category': 'Preparation Support',
            'priority': 'medium',
            'message': 'Consider enrolling in a review center for structured guidance.',
            'action': 'Research accredited review centers in your area'
        })
    
    # Confidence level recommendation
    if input_data['Confidence'] < 5:
        recommendations.append({
            'category': 'Mental Preparation',
            'priority': 'medium',
            'message': 'Low confidence can impact performance. Build confidence through consistent practice.',
            'action': 'Set small achievable goals and track your progress'
        })
    
    # Test anxiety recommendation
    if input_data['TestAnxiety'] > 7:
        recommendations.append({
            'category': 'Mental Health',
            'priority': 'high',
            'message': 'High test anxiety can significantly affect performance. Consider stress management techniques.',
            'action': 'Practice relaxation techniques, meditation, or seek counseling'
        })
    
    # English proficiency recommendation
    if input_data['EnglishProficiency'] < 5:
        recommendations.append({
            'category': 'Language Skills',
            'priority': 'high',
            'message': 'Low English proficiency may affect exam comprehension. Improve your language skills.',
            'action': 'Practice reading English materials and take language courses'
        })
    
    # Motivation recommendation
    if input_data['MotivationScore'] < 5:
        recommendations.append({
            'category': 'Motivation',
            'priority': 'medium',
            'message': 'Low motivation can hinder consistent study habits. Find ways to stay motivated.',
            'action': 'Set clear goals, join study groups, and remind yourself of your career aspirations'
        })
    
    # Social support recommendation
    if input_data['SocialSupport'] < 5:
        recommendations.append({
            'category': 'Support System',
            'priority': 'medium',
            'message': 'Limited social support can increase stress. Connect with supportive individuals.',
            'action': 'Join peer study groups and communicate your needs to family and friends'
        })
    
    # GPA-based recommendation
    if input_data['GPA'] > 2.5:
        recommendations.append({
            'category': 'Academic Foundation',
            'priority': 'medium',
            'message': 'Strengthen your foundational knowledge through comprehensive review.',
            'action': 'Review undergraduate materials and focus on core concepts'
        })
    
    return recommendations

def get_risk_level(probability):
    """
    Determine risk level based on pass probability
    """
    if probability >= 0.80:
        return {
            'level': 'Very Low Risk',
            'color': 'success',
            'description': 'High likelihood of passing. Maintain your current preparation.'
        }
    elif probability >= 0.70:
        return {
            'level': 'Low Risk',
            'color': 'info',
            'description': 'Good chance of passing. Continue with focused preparation.'
        }
    elif probability >= 0.60:
        return {
            'level': 'Moderate Risk',
            'color': 'warning',
            'description': 'Borderline probability. Increase study efforts and address weak areas.'
        }
    else:
        return {
            'level': 'High Risk',
            'color': 'danger',
            'description': 'Low probability of passing. Significant improvement needed in preparation.'
        }