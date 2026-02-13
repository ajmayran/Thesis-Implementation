# PREDICTING SUCCESS OF FIRST-TIME TAKERS IN SOCIAL WORK LICENSURE EXAMINATION USING ENSEMBLE TECHNIQUES

This repository contains the implementation for our thesis project, which aims to predict the success of first-time takers in the Social Work Licensure Examination using ensemble machine learning techniques.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation & Setup](#installation--setup)
- [Members](#members)
- [License](#license)

## Project Overview
This project explores and develops predictive models using ensemble techniques to forecast the outcomes of first-time takers in the Social Work Licensure Examination. The goal is to provide insights that can help educators and institutions better support examinees.

## Technologies Used
- Python
- Django 
- scikit-learn
- pandas / numpy
- matplotlib / seaborn
- joblib
- whitenoise (static file serving)


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
venv\Scripts\Activate
```


### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```


### 7. Apply Database Migrations
```bash
python manage.py makemigrations
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


## Members
- Alvan Jay Mayran
- Cyrus Bon Dimain
- Gina-Lenn Bejoc

## License
