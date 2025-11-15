from django.core.management.base import BaseCommand
from prediction.models import RegressionModelPerformance
import sys
import os
import numpy as np

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Command(BaseCommand):
    help = 'Train regression models and save performance metrics to database'

    def handle(self, *args, **options):
        try:
            self.stdout.write(self.style.SUCCESS('Starting regression model training...'))
            
            X_train = np.load('models/processed_data/X_train.npy')
            X_test = np.load('models/processed_data/X_test.npy')
            y_train = np.load('models/processed_data/y_train.npy')
            y_test = np.load('models/processed_data/y_test.npy')
            
            X_full = np.vstack([X_train, X_test])
            y_full = np.concatenate([y_train, y_test])
            
            cv = KFold(n_splits=10, shuffle=True, random_state=42)
            
            model_configs = {
                'knn': {
                    'model': KNeighborsRegressor(),
                    'params': {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance']
                    }
                },
                'decision_tree': {
                    'model': DecisionTreeRegressor(random_state=42),
                    'params': {
                        'max_depth': [3, 5, 7, 10, None],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'random_forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100],
                        'max_depth': [5, 10, 15, None]
                    }
                },
                'svr': {
                    'model': SVR(),
                    'params': {
                        'C': [0.1, 1.0, 10.0],
                        'epsilon': [0.01, 0.1, 0.2]
                    }
                },
                'ridge': {
                    'model': Ridge(),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0, 100.0]
                    }
                }
            }
            
            RegressionModelPerformance.objects.filter(model_type='regression').update(is_active=False)
            
            for name, config in model_configs.items():
                self.stdout.write(f'Training {name.upper()} with 10-fold CV...')
                
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                grid_search.fit(X_full, y_full)
                best_model = grid_search.best_estimator_
                
                cv_scores = cross_val_score(
                    best_model, X_full, y_full,
                    cv=cv, scoring='neg_mean_squared_error'
                )
                cv_mse = -cv_scores.mean()
                cv_rmse = np.sqrt(cv_mse)
                cv_std = np.std(cv_scores)
                
                y_pred = best_model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_pred)
                test_rmse = np.sqrt(test_mse)
                test_mae = mean_absolute_error(y_test, y_pred)
                test_r2 = r2_score(y_test, y_pred)
                
                RegressionModelPerformance.objects.create(
                    model_name=name,
                    model_type='regression',
                    rmse=test_rmse,
                    mae=test_mae,
                    r2_score=test_r2,
                    mse=test_mse,
                    cv_rmse=cv_rmse,
                    cv_std=cv_std,
                    is_active=True
                )
                
                self.stdout.write(self.style.SUCCESS(
                    f'Saved {name}: RMSE={test_rmse:.4f}, R²={test_r2:.4f}, '
                    f'CV_RMSE={cv_rmse:.4f}±{cv_std:.4f}'
                ))
            
            best_rmse = RegressionModelPerformance.objects.filter(
                model_type='regression', is_active=True
            ).order_by('rmse').first()
            
            self.stdout.write(self.style.SUCCESS(
                f'\nRegression training completed!'
                f'\nBest model: {best_rmse.model_name} (RMSE: {best_rmse.rmse:.4f})'
                f'\nTotal models trained: {len(model_configs)}'
            ))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error training regression models: {str(e)}'))
            import traceback
            traceback.print_exc()