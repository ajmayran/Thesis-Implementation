from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError

User = get_user_model()

class LoginForm(forms.Form):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Email',
            'autofocus': True,
            'id': 'id_email'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Password',
            'id': 'id_password'
        })
    )
    remember_me = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'id': 'id_remember_me'
        })
    )
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        return email

class RegistrationForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Email',
            'id': 'id_email'
        })
    )
    first_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'First Name',
            'id': 'id_first_name'
        })
    )
    last_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Last Name',
            'id': 'id_last_name'
        })
    )
    phone_number = forms.CharField(
        max_length=15,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Phone Number (Optional)',
            'id': 'id_phone_number'
        })
    )
    student_id = forms.CharField(
        max_length=9,
        min_length=9,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Student ID (9 digits)',
            'id': 'id_student_id'
        })
    )
    
    class Meta:
        model = User
        fields = ('email', 'first_name', 'last_name', 
                  'phone_number', 'student_id', 'password1', 'password2')
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Password',
            'id': 'id_password1'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirm Password',
            'id': 'id_password2'
        })
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if email:
            if User.objects.filter(email=email).exists():
                raise ValidationError('This email is already registered.')
        return email
    
    def clean_student_id(self):
        student_id = self.cleaned_data.get('student_id')
        if student_id:
            if not student_id.isdigit():
                raise ValidationError('Student ID must contain only numbers.')
            if len(student_id) != 9:
                raise ValidationError('Student ID must be exactly 9 digits.')
            if User.objects.filter(student_id=student_id).exists():
                raise ValidationError('This Student ID is already registered.')
        return student_id
    
    def save(self, commit=True):
        user = super().save(commit=False)
        base_username = self.cleaned_data['email'].split('@')[0]
        username = base_username
        suffix = 1
        while User.objects.filter(username=username).exists():
            username = f"{base_username}{suffix}"
            suffix += 1
        user.username = username
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.phone_number = self.cleaned_data.get('phone_number', '')
        user.student_id = self.cleaned_data['student_id']
        user.role = 'student'
        
        if commit:
            user.save()
        return user