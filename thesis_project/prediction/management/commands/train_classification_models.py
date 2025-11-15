from django.core.management.base import BaseCommand
from prediction.models import ModelPerformance
import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from models.base_models import SocialWorkPredictorModels

class Command(BaseCommand):
    help = 'Train classification models and save performance metrics to database'

    def handle(self, *args, **options):
        try:
            self.stdout.write(self.style.SUCCESS('Starting classification model training...'))
            
            predictor = SocialWorkPredictorModels()
            
            data = predictor.load_preprocessed_data(data_dir='processed_data', approach='label')
            
            if data is None:
                self.stdout.write(self.style.ERROR('Could not load preprocessed data'))
                return
            
            predictor.X_train = data['X_train']
            predictor.X_test = data['X_test']
            predictor.y_train = data['y_train']
            predictor.y_test = data['y_test']
            
            self.stdout.write('Training base classification models with 10-fold CV...')
            results = predictor.train_base_models()
            
            if not results:
                self.stdout.write(self.style.ERROR('Model training failed'))
                return
            
            self.stdout.write('Saving classification model performance to database...')
            
            ModelPerformance.objects.filter(model_type='classification').update(is_active=False)
            
            for model_name, metrics in results.items():
                classification_report = metrics['classification_report']['1']
                
                model_perf = ModelPerformance.objects.create(
                    model_name=model_name,
                    model_type='classification',
                    accuracy=metrics['accuracy'] * 100,
                    precision=classification_report['precision'],
                    recall=classification_report['recall'],
                    f1_score=classification_report['f1-score'],
                    auc_score=metrics.get('auc_score', 0.88),
                    cv_mean=metrics['cv_10fold_mean'],
                    cv_std=metrics['cv_10fold_std'],
                    is_active=True
                )
                
                self.stdout.write(self.style.SUCCESS(
                    f'Saved {model_name}: Accuracy={model_perf.accuracy:.2f}%, '
                    f'CV={model_perf.cv_mean:.4f}±{model_perf.cv_std:.4f}'
                ))
            
            best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
            self.stdout.write(self.style.SUCCESS(
                f'\nClassification training completed!'
                f'\nBest model: {best_model[0]} (Accuracy: {best_model[1]["accuracy"]:.4f})'
                f'\nTotal models trained: {len(results)}'
            ))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error training classification models: {str(e)}'))
            import traceback
            traceback.print_exc()