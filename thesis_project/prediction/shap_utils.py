import os
import pandas as pd
from django.conf import settings
from .config import SELECTED_MODEL_NAME

# Build SHAP data path dynamically based on active model
SHAP_DATA_PATH = os.path.join(
    settings.BASE_DIR.parent, 
    'models', 
    'saved_ensemble_models', 
    'shap_analysis', 
    f'{SELECTED_MODEL_NAME}_shap_importance.csv'
)

def load_shap_importance():
    """
    Load SHAP feature importance from CSV file for the active model
    Returns a dictionary mapping feature names to their importance scores
    """
    try:
        if not os.path.exists(SHAP_DATA_PATH):
            print(f"[WARNING] SHAP importance file not found at {SHAP_DATA_PATH}")
            return get_default_importance()
        
        df = pd.read_csv(SHAP_DATA_PATH)
        
        # Create dictionary mapping feature to importance
        importance_dict = dict(zip(df['Feature'], df['SHAP_Importance']))
        
        # Sort by importance descending
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        print(f"[INFO] Loaded SHAP importance for {len(importance_dict)} features from model: {SELECTED_MODEL_NAME}")
        return importance_dict
    
    except Exception as e:
        print(f"[ERROR] Failed to load SHAP importance: {str(e)}")
        return get_default_importance()


def get_default_importance():
    """
    Fallback importance values if SHAP data cannot be loaded
    """
    return {
        'StudyHours': 0.3607,
        'GPA': 0.2978,
        'ReviewCenter': 0.2658,
        'Age': 0.2646,
        'TestAnxiety': 0.2575,
        'SleepHours': 0.2167,
        'EmploymentStatus': 0.2139,
        'EnglishProficiency': 0.1600,
        'IncomeLevel': 0.1018,
        'Scholarship': 0.0943,
        'MockExamScore': 0.0192,
        'Confidence': 0.0173,
        'SocialSupport': 0.0138,
        'MotivationScore': 0.0054,
        'Gender': 0.0,
        'InternshipGrade': 0.0
    }


def get_feature_importance_ranking():
    """
    Get ranked list of features by importance
    Returns list of tuples (feature_name, importance_score, rank)
    """
    importance_dict = load_shap_importance()
    
    sorted_features = sorted(
        importance_dict.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    ranked_features = [
        (feature, importance, rank + 1) 
        for rank, (feature, importance) in enumerate(sorted_features)
    ]
    
    return ranked_features


def get_top_features(n=5):
    """
    Get top N most important features
    """
    ranked = get_feature_importance_ranking()
    return ranked[:n]


def get_feature_impact_category(importance_score):
    """
    Categorize feature impact based on importance score
    """
    if importance_score >= 0.25:
        return {
            'category': 'Critical',
            'color': '#ef4444',
            'description': 'Has major impact on exam results'
        }
    elif importance_score >= 0.15:
        return {
            'category': 'High',
            'color': '#f59e0b',
            'description': 'Significantly affects exam performance'
        }
    elif importance_score >= 0.05:
        return {
            'category': 'Moderate',
            'color': '#3b82f6',
            'description': 'Moderately influences exam outcomes'
        }
    else:
        return {
            'category': 'Low',
            'color': '#10b981',
            'description': 'Minor influence on exam results'
        }


def analyze_user_weaknesses(input_data):
    """
    Analyze user's weak areas based on SHAP importance
    Returns list of features that need improvement
    """
    importance_dict = load_shap_importance()
    weaknesses = []
    
    thresholds = {
        'StudyHours': 20,
        'GPA': 2.5,
        'InternshipGrade': 70,
        'MotivationScore': 5,
        'EnglishProficiency': 5,
        'SleepHours': 6,
        'MockExamScore': 70,
        'Confidence': 5,
        'TestAnxiety': 5,
        'SocialSupport': 5
    }
    
    for feature, threshold in thresholds.items():
        if feature in input_data and feature in importance_dict:
            value = input_data[feature]
            importance = importance_dict[feature]
            
            if feature == 'TestAnxiety':
                if value > threshold and importance > 0.01:
                    weaknesses.append({
                        'feature': feature,
                        'value': value,
                        'threshold': threshold,
                        'importance': importance,
                        'impact': get_feature_impact_category(importance),
                        'display_name': get_feature_display_name(feature)
                    })
            else:
                if value < threshold and importance > 0.01:
                    weaknesses.append({
                        'feature': feature,
                        'value': value,
                        'threshold': threshold,
                        'importance': importance,
                        'impact': get_feature_impact_category(importance),
                        'display_name': get_feature_display_name(feature)
                    })
    
    weaknesses.sort(key=lambda x: x['importance'], reverse=True)
    return weaknesses


def analyze_user_strengths(input_data):
    """
    Analyze user's strong areas based on SHAP importance
    """
    importance_dict = load_shap_importance()
    strengths = []
    
    excellence_thresholds = {
        'StudyHours': 30,
        'GPA': 3.5,
        'InternshipGrade': 85,
        'MotivationScore': 8,
        'EnglishProficiency': 8,
        'SleepHours': 7,
        'MockExamScore': 85,
        'Confidence': 8,
        'TestAnxiety': 3,
        'SocialSupport': 8
    }
    
    for feature, threshold in excellence_thresholds.items():
        if feature in input_data and feature in importance_dict:
            value = input_data[feature]
            importance = importance_dict[feature]
            
            if feature == 'TestAnxiety':
                if value <= threshold and importance > 0.01:
                    strengths.append({
                        'feature': feature,
                        'value': value,
                        'importance': importance,
                        'impact': get_feature_impact_category(importance),
                        'display_name': get_feature_display_name(feature)
                    })
            else:
                if value >= threshold and importance > 0.01:
                    strengths.append({
                        'feature': feature,
                        'value': value,
                        'importance': importance,
                        'impact': get_feature_impact_category(importance),
                        'display_name': get_feature_display_name(feature)
                    })
    
    strengths.sort(key=lambda x: x['importance'], reverse=True)
    return strengths


def get_feature_display_name(feature_name):
    """
    Convert feature name to human-readable display name
    """
    display_names = {
        'StudyHours': 'Study Hours per Week',
        'InternshipGrade': 'Internship Grade',
        'GPA': 'GPA',
        'MotivationScore': 'Motivation Score',
        'EmploymentStatus': 'Employment Status',
        'IncomeLevel': 'Income Level',
        'EnglishProficiency': 'English Proficiency',
        'Age': 'Age',
        'TestAnxiety': 'Test Anxiety Level',
        'SleepHours': 'Sleep Hours per Night',
        'MockExamScore': 'Mock Exam Score',
        'ReviewCenter': 'Review Center Attendance',
        'Confidence': 'Confidence Level',
        'SocialSupport': 'Social Support',
        'Scholarship': 'Scholarship Status',
        'Gender': 'Gender'
    }
    return display_names.get(feature_name, feature_name)


def get_personalized_insights(input_data, pass_probability):
    """
    Generate personalized insights based on SHAP importance and user data
    """
    weaknesses = analyze_user_weaknesses(input_data)
    strengths = analyze_user_strengths(input_data)
    top_features = get_top_features(5)
    
    insights = {
        'weaknesses': weaknesses[:3],
        'strengths': strengths[:3],
        'top_impact_features': [
            {
                'name': feature,
                'display_name': get_feature_display_name(feature),
                'importance': importance,
                'impact': get_feature_impact_category(importance),
                'user_value': input_data.get(feature)
            }
            for feature, importance, rank in top_features
        ],
        'priority_areas': get_priority_improvement_areas(weaknesses, top_features)
    }
    
    return insights


def get_priority_improvement_areas(weaknesses, top_features):
    """
    Identify which weak areas should be prioritized based on importance
    """
    top_feature_names = [f[0] for f in top_features]
    
    priority = []
    for weakness in weaknesses:
        if weakness['feature'] in top_feature_names:
            priority.append({
                'feature': weakness['display_name'],
                'importance': weakness['importance'],
                'current_value': weakness['value'],
                'target_value': weakness['threshold'],
                'gap': abs(weakness['value'] - weakness['threshold']),
                'impact': weakness['impact']
            })
    
    return priority[:3]