import os
import json
import joblib
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from django.utils import timezone
from prediction.models import RegressionModelPerformance

class Command(BaseCommand):
    help = 'Save ensemble model performance metrics to database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--models-dir',
            type=str,
            default='models/saved_ensemble_models',
            help='Directory containing saved ensemble models'
        )

    def handle(self, *args, **options):
        models_dir = options['models_dir']
        
        self.stdout.write(self.style.SUCCESS('Starting ensemble metrics import...'))
        
        # Define ensemble models to process
        ensemble_models = [
            'bagging_random_forest',
            'boosting_gradient_boost',
            'stacking_ridge'
        ]
        
        saved_count = 0
        
        for model_name in ensemble_models:
            try:
                # Load prediction details CSV
                csv_path = os.path.join(models_dir, f'{model_name}_test_predictions.csv')
                
                if not os.path.exists(csv_path):
                    self.stdout.write(
                        self.style.WARNING(f'Predictions file not found: {csv_path}')
                    )
                    continue
                
                # Read predictions
                predictions_df = pd.read_csv(csv_path)
                
                # Calculate metrics
                errors = predictions_df['Error'].values
                absolute_errors = predictions_df['Absolute_Error'].values
                actual = predictions_df['Actual'].values
                predicted = predictions_df['Predicted'].values
                
                # Calculate performance metrics
                mse = np.mean(errors ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(absolute_errors)
                
                # Calculate R² score
                ss_res = np.sum(errors ** 2)
                ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                r2_score = 1 - (ss_res / ss_tot)
                
                # Try to load additional metrics from model metadata if available
                cv_rmse = None
                cv_std = None
                
                # Check for metadata file
                metadata_path = os.path.join(models_dir, f'{model_name}_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        cv_rmse = metadata.get('cv_rmse')
                        cv_std = metadata.get('cv_std')
                
                # Save to database
                performance, created = RegressionModelPerformance.objects.update_or_create(
                    model_name=model_name,
                    model_type='ensemble_regression',
                    trained_at=timezone.now(),
                    defaults={
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'r2_score': float(r2_score),
                        'mse': float(mse),
                        'cv_rmse': float(cv_rmse) if cv_rmse else None,
                        'cv_std': float(cv_std) if cv_std else None,
                        'is_active': True
                    }
                )
                
                action = 'Created' if created else 'Updated'
                self.stdout.write(
                    self.style.SUCCESS(
                        f'{action} {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2_score:.4f}'
                    )
                )
                
                saved_count += 1
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error processing {model_name}: {str(e)}')
                )
        
        self.stdout.write(
            self.style.SUCCESS(f'\nSuccessfully saved {saved_count} ensemble model metrics')
        )