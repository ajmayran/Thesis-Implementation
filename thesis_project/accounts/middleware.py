from django.shortcuts import redirect
from django.urls import reverse

class LoginRequiredMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.exempt_urls = [
            reverse('accounts:login'),
            reverse('accounts:register'),
        ]
    
    def __call__(self, request):
        if not request.user.is_authenticated:
            path = request.path_info
            
            if not any(path.startswith(url) for url in self.exempt_urls):
                if not path.startswith('/static/') and not path.startswith('/media/'):
                    return redirect('accounts:login')
        
        response = self.get_response(request)
        return response