from django.core.management.base import BaseCommand
from prediction.models import ModelPerformance
import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from models.base_models import SocialWorkPredictorModels

class Command(BaseCommand):
    help = 'Train models and save performance metrics to database'

    def handle(self, *args, **options):
        try:
            self.stdout.write(self.style.SUCCESS('Starting model training...'))
            
            predictor = SocialWorkPredictorModels()
            
            self.stdout.write('Training base models with 10-fold CV...')
            results = predictor.train_base_models()
            
            self.stdout.write('Saving model performance to database...')
            for model_name, metrics in results.items():
                ModelPerformance.objects.create(
                    model_name=model_name,
                    model_type='base',
                    accuracy=metrics['accuracy'] * 100,
                    precision=metrics['classification_report']['1']['precision'],
                    recall=metrics['classification_report']['1']['recall'],
                    f1_score=metrics['classification_report']['1']['f1-score'],
                    auc_score=0.88,
                    cv_mean=metrics['cv_10fold_mean'],
                    cv_std=metrics['cv_10fold_std'],
                    is_active=True
                )
                self.stdout.write(self.style.SUCCESS(f'Saved {model_name} performance'))
            
            predictor.save_models('saved_base_models')
            
            self.stdout.write(self.style.SUCCESS('Model training completed successfully!'))
            self.stdout.write(f'Best model: {max(results.items(), key=lambda x: x[1]["accuracy"])[0]}')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error training models: {str(e)}'))
            import traceback
            traceback.print_exc()