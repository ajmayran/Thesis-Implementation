import sys
import os
import numpy as np
import joblib
import pandas as pd

# Add parent directory to path for model imports
parent_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..',
    '..'
))

if parent_path not in sys.path:
    sys.path.insert(0, parent_path)


def prepare_input_data(form_data):
    """
    Prepare input data dictionary from form data for prediction
    
    Args:
        form_data: Dictionary containing form input values
    
    Returns:
        dict: Prepared input data dictionary
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
    Generate personalized recommendations based on input data and prediction.
    Returns list of user-friendly text recommendations for social work graduates.
    
    Args:
        input_data: Dictionary containing user's input data
        pass_probability: Predicted probability of passing (0-1 range)
    
    Returns:
        list: List of recommendation strings
    """
    recommendations = []
    
    # Convert probability to percentage for internal calculations
    pass_probability_pct = pass_probability * 100
    
    # Study hours recommendation
    if input_data['StudyHours'] < 5:
        recommendations.append(
            "Increase your daily study hours to at least 6-8 hours of focused study. "
            "Create a structured study schedule that balances all exam topics including "
            "social work theories, case management, ethics, and social welfare policies."
        )
    elif input_data['StudyHours'] < 7:
        recommendations.append(
            "Your study hours are moderate. Consider adding 1-2 more hours daily to strengthen "
            "your preparation, especially in challenging areas."
        )
    
    # Sleep hours recommendation
    if input_data['SleepHours'] < 6:
        recommendations.append(
            "Get adequate sleep of 7-8 hours per night. Insufficient sleep can significantly "
            "affect your cognitive performance, memory retention, and concentration during the exam. "
            "Quality sleep is as important as study time."
        )
    elif input_data['SleepHours'] > 9:
        recommendations.append(
            "While rest is important, excessive sleep may indicate fatigue or lack of study routine. "
            "Aim for 7-8 hours of quality sleep and use the extra time for focused study."
        )
    
    # Mock exam recommendation
    if input_data.get('MockExamScore') and input_data['MockExamScore'] < 70:
        recommendations.append(
            "Your mock exam scores indicate significant room for improvement. Take more practice tests, "
            "carefully review all incorrect answers, identify patterns in your mistakes, and focus on "
            "strengthening weak areas in social work knowledge."
        )
    elif input_data.get('MockExamScore') and input_data['MockExamScore'] < 80:
        recommendations.append(
            "Your mock exam performance is fair but needs improvement. Continue taking practice tests "
            "and aim for consistent scores above 80% to feel confident on exam day."
        )
    
    # Review center recommendation
    if not input_data['ReviewCenter'] and pass_probability < 0.7:
        recommendations.append(
            "Consider enrolling in an accredited review center for social work licensure. "
            "Professional guidance, structured review programs, and comprehensive materials "
            "can significantly improve your exam preparation and boost your confidence."
        )
    elif not input_data['ReviewCenter']:
        recommendations.append(
            "While you're preparing independently, consider supplementing with review materials "
            "from accredited review centers or joining online review programs for additional support."
        )
    
    # Confidence level recommendation
    if input_data['Confidence'] < 5:
        recommendations.append(
            "Build your confidence through consistent practice and preparation. Set small, "
            "achievable study goals, track your progress daily, celebrate small wins, and "
            "remind yourself of your capabilities. Self-belief is crucial for exam success."
        )
    elif input_data['Confidence'] > 8 and pass_probability < 0.7:
        recommendations.append(
            "While confidence is good, ensure it's backed by thorough preparation. "
            "Take regular mock exams to validate your readiness and avoid overconfidence."
        )
    
    # Test anxiety recommendation
    if input_data['TestAnxiety'] > 7:
        recommendations.append(
            "High test anxiety can negatively impact your performance on exam day. "
            "Practice relaxation techniques such as deep breathing exercises, meditation, "
            "progressive muscle relaxation, or mindfulness. Consider seeking support from "
            "counselors or mental health professionals if anxiety persists."
        )
    elif input_data['TestAnxiety'] > 5:
        recommendations.append(
            "Moderate test anxiety is normal. Practice stress management techniques and "
            "familiarize yourself with the exam format through mock tests to reduce anxiety."
        )
    
    # English proficiency recommendation
    if input_data['EnglishProficiency'] < 5:
        recommendations.append(
            "Improve your English language skills as the exam requires good reading comprehension. "
            "Read English materials regularly (journals, articles, case studies), practice answering "
            "questions in English, and consider taking language enhancement courses or tutorials."
        )
    elif input_data['EnglishProficiency'] < 7:
        recommendations.append(
            "Your English proficiency is moderate. Continue reading social work literature in English "
            "and practice articulating social work concepts clearly."
        )
    
    # Motivation recommendation
    if input_data['MotivationScore'] < 5:
        recommendations.append(
            "Stay motivated throughout your preparation journey. Set clear career goals, "
            "visualize your success as a licensed social worker, join study groups for peer support, "
            "connect with licensed social workers for inspiration, and regularly remind yourself "
            "why you chose this noble profession."
        )
    elif input_data['MotivationScore'] < 7:
        recommendations.append(
            "Maintain your motivation by setting weekly goals and rewarding yourself for achievements. "
            "Connect with fellow reviewees to stay encouraged."
        )
    
    # Social support recommendation
    if input_data['SocialSupport'] < 5:
        recommendations.append(
            "Build a strong support system around you. Connect with fellow reviewees, "
            "join study groups or online communities, communicate your needs clearly to family "
            "and friends, and don't hesitate to ask for help when needed. A good support network "
            "can make a significant difference in your exam preparation."
        )
    elif input_data['SocialSupport'] < 7:
        recommendations.append(
            "Your support system is moderate. Consider expanding your network by joining "
            "review groups or online study communities for additional encouragement."
        )
    
    # GPA-based recommendation
    if input_data['GPA'] > 2.5:
        recommendations.append(
            "Strengthen your foundational knowledge by thoroughly reviewing your undergraduate materials. "
            "Focus on core social work concepts, theories (systems theory, ecological perspective, "
            "strengths-based approach), intervention methods, research methods, and professional ethics "
            "that are essential for the licensure examination."
        )
    elif input_data['GPA'] > 2.0:
        recommendations.append(
            "Your undergraduate performance was good, but comprehensive review is still essential. "
            "Focus on areas where you had challenges during college."
        )
    
    # Internship grade recommendation
    if input_data['InternshipGrade'] > 2.5:
        recommendations.append(
            "Your internship performance suggests you need to strengthen practical application knowledge. "
            "Review case studies, practice scenarios, and ethical dilemmas commonly encountered in social work."
        )
    
    # Employment status recommendation
    if input_data['EmploymentStatus'] == 'Employed Full-time':
        recommendations.append(
            "Balancing full-time work and exam preparation is challenging. Create a realistic study "
            "schedule that accommodates your work commitments. Wake up earlier or study in the evening "
            "in focused blocks. Consider requesting study leave or reduced hours closer to the exam date "
            "if possible. Maximize weekends for intensive review."
        )
    elif input_data['EmploymentStatus'] == 'Employed Part-time':
        recommendations.append(
            "Use your flexible schedule wisely. Dedicate your non-working days to intensive study "
            "and maintain a consistent daily review routine."
        )
    
    # Income level recommendation
    if input_data['IncomeLevel'] == 'Low':
        recommendations.append(
            "While financial constraints can be challenging, there are free online resources, "
            "study groups, and library materials available. Consider applying for review center scholarships "
            "or financial assistance programs offered by professional organizations."
        )
    
    # General recommendations based on probability
    if pass_probability_pct >= 80:
        recommendations.append(
            "Excellent work! You're on the right track. Continue with your current preparation strategy "
            "and maintain consistency. Focus on retaining what you've learned through regular review, "
            "practice time management for the actual exam, and stay calm and confident. Your preparation "
            "level is strong."
        )
    elif pass_probability_pct >= 70:
        recommendations.append(
            "You have a good foundation and fair chance of passing. To improve further, increase your "
            "study intensity by 1-2 hours daily, take more comprehensive practice tests, identify and "
            "address any remaining weak areas, and maintain your current momentum."
        )
    elif pass_probability_pct >= 60:
        recommendations.append(
            "You are at a borderline level. Your preparation needs significant improvement. "
            "Intensify your study efforts immediately, consider enrolling in an intensive review program, "
            "take multiple mock exams weekly, and focus heavily on weak areas. Consider whether "
            "you need more preparation time."
        )
    else:
        recommendations.append(
            "Your current preparation level needs substantial improvement. We strongly recommend "
            "postponing the exam to the next available schedule if possible to allow adequate preparation time. "
            "Enroll in a reputable review center, dedicate full-time hours to studying if feasible, "
            "seek mentorship from licensed social workers, and create a comprehensive 3-4 month study plan."
        )
    
    # Final encouragement
    recommendations.append(
        "Remember, becoming a licensed social worker is a journey. Stay committed to your goal, "
        "maintain a positive mindset, take care of your physical and mental health, and believe in yourself. "
        "Your dedication to serving communities and helping others is admirable. Good luck!"
    )
    
    return recommendations

def get_risk_level(probability):
    """
    Determine risk level based on pass probability with user-friendly descriptions.
    
    Args:
        probability: Pass probability (0-1 range)
    
    Returns:
        dict: Dictionary containing risk level information
    """
    # Convert to percentage for easier comparison
    probability_pct = probability * 100
    
    if probability_pct >= 80:
        return {
            'level': 'low',
            'color': 'green',
            'icon': 'circle-check',
            'message': 'Excellent preparation level! You have a very high likelihood of passing the Social Work '
                      'Licensure Examination. Your current study habits and preparation strategy are working well. '
                      'Keep up your current routine, stay consistent, and maintain your confidence. You are well-prepared!'
        }
    elif probability_pct >= 70:
        return {
            'level': 'medium-low',
            'color': 'blue',
            'icon': 'circle-info',
            'message': 'Good preparation level! You have a solid chance of passing the examination. '
                      'Your foundation is strong, but there is still room for improvement. Continue with focused '
                      'study, practice regularly with mock exams, and strengthen any weak areas. With continued '
                      'effort, you can increase your chances significantly.'
        }
    elif probability_pct >= 60:
        return {
            'level': 'medium',
            'color': 'yellow',
            'icon': 'triangle-exclamation',
            'message': 'Moderate risk level - You are at the borderline. While you have a chance of passing, '
                      'your preparation needs improvement. Increase your study efforts significantly, '
                      'address weak areas immediately, consider enrolling in a review center for additional support, '
                      'and take more practice tests. The next few weeks of preparation are critical.'
        }
    elif probability_pct >= 50:
        return {
            'level': 'medium-high',
            'color': 'orange',
            'icon': 'exclamation',
            'message': 'High risk level - Your current preparation is insufficient for a comfortable pass. '
                      'Immediate and intensive action is needed. Consider postponing the exam if possible to allow '
                      'more preparation time. Enroll in an intensive review program, dedicate significantly more hours '
                      'to studying, and seek help from mentors or review centers.'
        }
    else:
        return {
            'level': 'high',
            'color': 'red',
            'icon': 'circle-exclamation',
            'message': 'Critical risk level - Your current preparation needs substantial improvement. '
                      'We strongly recommend postponing the examination to the next schedule to allow adequate '
                      'preparation time. Enroll in a comprehensive review center program, create a structured '
                      '3-4 month study plan, seek mentorship from licensed social workers, and dedicate full-time '
                      'equivalent hours to preparation. Success is possible with proper preparation time and effort.'
        }