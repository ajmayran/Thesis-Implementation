from django import forms

class PredictionForm(forms.Form):
    """Form for collecting student data for prediction"""
    
    # Personal Information
    age = forms.IntegerField(
        label='Age',
        min_value=21,
        max_value=65,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter age',
            'required': True
        })
    )
    
    gender = forms.ChoiceField(
        label='Gender',
        choices=[('', 'Select Gender'), ('Male', 'Male'), ('Female', 'Female')],
        widget=forms.Select(attrs={
            'class': 'form-control',
            'required': True
        })
    )
    
    # Academic Information
    gpa = forms.FloatField(
        label='GPA (Grade Point Average)',
        min_value=1.0,
        max_value=4.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter GPA',
            'step': '0.01',
            'required': True
        }),
        help_text='Your cumulative GPA'
    )
    
    internship_grade = forms.FloatField(
        label='Internship Grade',
        min_value=1.0,
        max_value=3.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter internship grade (1.0-3.0)',
            'step': '0.25',
            'required': True
        }),
        help_text='Your field internship grade'
    )
    
    scholarship = forms.ChoiceField(
        label='Scholarship Holder',
        choices=[('', 'Select'), ('0', 'No'), ('1', 'Yes')],
        widget=forms.Select(attrs={
            'class': 'form-control',
            'required': True
        }),
        help_text='Do you have a scholarship?'
    )
    
    # Study Habits
    study_hours = forms.IntegerField(
        label='Daily Study Hours',
        min_value=0,
        max_value=24,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter daily study hours (0-24)',
            'required': True
        }),
        help_text='Average hours spent studying per day'
    )
    
    sleep_hours = forms.IntegerField(
        label='Daily Sleep Hours',
        min_value=0,
        max_value=24,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter daily sleep hours (0-24)',
            'required': True
        }),
        help_text='Average hours of sleep per night'
    )
    
    review_center = forms.ChoiceField(
        label='Review Center Attendance',
        choices=[('', 'Select'), ('0', 'No'), ('1', 'Yes')],
        widget=forms.Select(attrs={
            'class': 'form-control',
            'required': True
        }),
        help_text='Are you attending a review center?'
    )
    
    # Preparedness Indicators
    confidence = forms.IntegerField(
        label='Confidence Level',
        min_value=1,
        max_value=5,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Rate 1-5 (1=Low, 5=High)',
            'required': True
        }),
        help_text='How confident are you about passing? (1-5 scale)'
    )
    
    mock_exam_score = forms.FloatField(
        label='Mock Exam Score (%)',
        min_value=0,
        max_value=100,
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter mock exam score (optional)',
            'step': '0.01'
        }),
        help_text='Your latest mock/practice exam score (optional)'
    )
    
    # Socioeconomic Factors
    income_level = forms.ChoiceField(
        label='Family Income Level',
        choices=[
            ('', 'Select Income Level'),
            ('Low', 'Low'),
            ('Middle', 'Middle'),
            ('High', 'High')
        ],
        widget=forms.Select(attrs={
            'class': 'form-control',
            'required': True
        })
    )
    
    employment_status = forms.ChoiceField(
        label='Employment Status',
        choices=[
            ('', 'Select Employment Status'),
            ('Unemployed', 'Unemployed'),
            ('Unskilled', 'Unskilled Worker'),
            ('Skilled', 'Skilled Worker'),
            ('Professional', 'Professional')
        ],
        widget=forms.Select(attrs={
            'class': 'form-control',
            'required': True
        }),
        help_text='Current employment status'
    )
    
    def clean(self):
        """Additional validation"""
        cleaned_data = super().clean()
        
        # Validate study + sleep hours <= 24
        study_hours = cleaned_data.get('study_hours', 0)
        sleep_hours = cleaned_data.get('sleep_hours', 0)
        
        if study_hours + sleep_hours > 24:
            raise forms.ValidationError(
                'Total study hours and sleep hours cannot exceed 24 hours per day.'
            )
        
        return cleaned_data