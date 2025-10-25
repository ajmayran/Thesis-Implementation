# PREDICTING SUCCESS OF FIRST-TIME TAKERS IN SOCIAL WORK LICENSURE EXAMINATION USING ENSEMBLE TECHNIQUES

This repository contains the implementation for our thesis project, which aims to predict the success of first-time takers in the Social Work Licensure Examination using ensemble machine learning techniques.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation & Setup](#installation--setup)
- [Troubleshooting](#troubleshooting)
- [Members](#members)
- [License](#license)

## Project Overview
This project explores and develops predictive models using ensemble techniques to forecast the outcomes of first-time takers in the Social Work Licensure Examination. The goal is to provide insights that can help educators and institutions better support examinees.

## Technologies Used
- Python
- Django / Django REST Framework
- scikit-learn
- pandas / numpy
- matplotlib / seaborn
- joblib
- whitenoise (static file serving)
- CORS headers


## Installation & Setup

### 1. Prerequisites
- Python 3.11+ (verify with `python --version` or `python3 --version`)
- Git


### 2. Clone the Repository
```bash
git clone https://github.com/ajmayran/Thesis-Implementation.git
cd Thesis-Implementation
```

### 3. (Recommended) Create & Activate a Virtual Environment
Mac/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Windows (CMD):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

Confirm:
```bash
which python        # mac/linux
# or
where python        # windows
```

### 4. Upgrade pip (optional but recommended)
```bash
pip install --upgrade pip
```

### 5. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 6. Environment Variables (If Needed)
If your Django settings rely on secrets (API keys, DEBUG flags, DB credentials), create a `.env` file (or export variables). Example:
```
DEBUG=True
SECRET_KEY=replace_me
ALLOWED_HOSTS=localhost,127.0.0.1
```
(Adjust according to how `settings.py` is implemented.)

### 7. Apply Database Migrations
```bash
python manage.py migrate
```

### 8. (Optional) Create a Superuser for Admin
```bash
python manage.py createsuperuser
```

### 9. Run the Development Server
```bash
python manage.py runserver
```


Run Tests (if configured):
```bash
python -m pytest
# or
python manage.py test
```


## Troubleshooting

| Issue | Possible Cause | Fix |
|-------|----------------|-----|
| ImportError / ModuleNotFoundError | Virtual environment not active | Activate venv or reinstall requirements |
| openpyxl version error | Typo in requirements (`=>`) | Change to `>=` and reinstall |
| Static files not served in production | Missing `collectstatic` or WhiteNoise config | Run `collectstatic`; verify `MIDDLEWARE` settings |
| CORS errors in browser | Missing/incorrect CORS config | Check `django-cors-headers` setup |

## Members
- Alvan Jay Mayran
- Cyrus Bon Dimain
- Gina-Lenn Bejoc

## License
