import os
from .settings import *
from .settings import BASE_DIR

ALLOWED_HOSTS = [os.environ['WEBSITE_HOSTNAME']]
SECRET_KEY = os.environ['SECRET_KEY']
CSRF_TRUSTED_ORIGINS = [f"https://{os.environ['WEBSITE_HOSTNAME']}"]
DEBUG = False


MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'accounts.middleware.LoginRequiredMiddleware',
]

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')



# connection_string = os.environ['AZURE_POSTGRE_CONNECTIONSTRING']
# # parameters = dict(pair.split('=') for pair in connection_string.split())
# parameters = dict(pair.split('=', 1) for pair in connection_string.split())

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.postgresql',
#         'NAME': parameters['DB_NAME'],
#         'USER': parameters['DB_USER'],
#         'PASSWORD': parameters['DB_PASSWORD'],
#         'HOST': parameters['DB_HOST'],
#     }
# }

connection_string = os.environ['AZURE_STORAGE_CONNECTION_STRING']

# Parse semicolon-separated key=value pairs
parameters = dict(pair.split('=', 1) for pair in connection_string.split(';') if '=' in pair)

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': parameters['DB_NAME'],
        'USER': parameters['DB_USER'],
        'PASSWORD': parameters['DB_PASSWORD'],
        'HOST': parameters['DB_HOST'],
        'PORT': '5432',
        'OPTIONS': {'sslmode': 'require'},
    }
}