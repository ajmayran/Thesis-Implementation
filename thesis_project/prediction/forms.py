from django import forms

class PredictionForm(forms.Form):
    # Personal Information
    age = forms.IntegerField(
        min_value=21,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your age'
        }),
        help_text='Your current age'
    )
    
    gender = forms.ChoiceField(
        choices=[
            ('', 'Select Gender'),
            ('Male', 'Male'),
            ('Female', 'Female')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    # Academic Background
    gpa = forms.FloatField(
        min_value=1.0,
        max_value=5.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 1.5',
            'step': '0.01'
        }),
        help_text='Your General Weighted Average'
    )
    
    internship_grade = forms.FloatField(
        min_value=1.0,
        max_value=5.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 1.75',
            'step': '0.01'
        }),
        help_text='Your internship grade (1.0 - 5.0)'
    )
    
    scholarship = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text='Check if you received any scholarship during your studies'
    )
    
    # Study Habits & Preparation
    study_hours = forms.IntegerField(
        min_value=0,
        max_value=24,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Hours per day'
        }),
        help_text='Average daily study hours for exam preparation'
    )
    
    sleep_hours = forms.IntegerField(
        min_value=0,
        max_value=24,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Hours per day'
        }),
        help_text='Average daily sleep hours'
    )
    
    review_center = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text='Check if you are enrolled in a review center'
    )
    
    confidence = forms.IntegerField(
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '1-10'
        }),
        help_text='Your confidence level about passing the exam (1=Low, 10=High)'
    )
    
    test_anxiety = forms.IntegerField(
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '1-10'
        }),
        help_text='Your test anxiety level (1=Low, 10=High)'
    )
    
    mock_exam_score = forms.FloatField(
        required=False,
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 75.5',
            'step': '0.1'
        }),
        help_text='Your most recent mock exam score (0-100)'
    )
    
    # New Fields
    english_proficiency = forms.IntegerField(
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '1-10'
        }),
        help_text='Your English proficiency level (1=Poor, 10=Excellent)'
    )
    
    motivation_score = forms.IntegerField(
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '1-10'
        }),
        help_text='Your motivation level for exam preparation (1=Low, 10=High)'
    )
    
    social_support = forms.IntegerField(
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '1-10'
        }),
        help_text='Level of support from family and friends (1=Low, 10=High)'
    )
    
    # Socioeconomic Background
    income_level = forms.ChoiceField(
        choices=[
            ('', 'Select Income Level'),
            ('Low', 'Low Income'),
            ('Middle', 'Middle Income'),
            ('High', 'High Income')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    employment_status = forms.ChoiceField(
        choices=[
            ('', 'Select Employment Status'),
            ('Unemployed', 'Unemployed'),
            ('Skilled', 'Skilled Worker'),
            ('Professional', 'Professional')
        ],
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Your current employment status'
    )