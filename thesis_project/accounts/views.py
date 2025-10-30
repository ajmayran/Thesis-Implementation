from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from .forms import LoginForm, RegistrationForm
from django.contrib.auth import get_user_model

User = get_user_model()

@csrf_protect
@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard:home')
    
    if request.method == 'POST':
        form = LoginForm(data=request.POST)
        
        if form.is_valid():
            email = form.cleaned_data.get('email')
            password = form.cleaned_data.get('password')
            remember_me = form.cleaned_data.get('remember_me')
            
            try:
                user = User.objects.get(email=email)
                user = authenticate(request, username=user.username, password=password)
                
                if user is not None:
                    if user.is_active:
                        login(request, user)
                        
                        if not remember_me:
                            request.session.set_expiry(0)
                        
                        messages.success(request, f'Welcome back, {user.first_name}!')
                        
                        next_url = request.GET.get('next')
                        if next_url:
                            return redirect(next_url)
                        
                        if user.is_admin():
                            return redirect('dashboard:home')
                        elif user.is_student():
                            return redirect('dashboard:home')
                        else:
                            return redirect('prediction:predict')
                    else:
                        messages.error(request, 'Your account has been disabled.')
                else:
                    messages.error(request, 'Invalid email or password.')
            except User.DoesNotExist:
                messages.error(request, 'Invalid email or password.')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = LoginForm()
    
    return render(request, 'pages/login.html', {'form': form})

@csrf_protect
@require_http_methods(["GET", "POST"])
def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard:home')
    
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        
        if form.is_valid():
            user = form.save()
            messages.success(request, 'Registration successful! Please login with your WMSU email.')
            return redirect('accounts:login')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = RegistrationForm()
    
    return render(request, 'pages/register.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('accounts:login')

@login_required
def profile_view(request):
    return render(request, 'pages/profile.html', {
        'user': request.user
    })