import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap
import warnings
from base_models import SocialWorkPredictorModels

warnings.filterwarnings('ignore')


class BaseModelsSHAPAnalysis:
    def __init__(self):
        self.models = {}
        self.shap_results = {}
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.X_test_subset = None
        self.y_test = None

    def load_data_and_models(
        self,
        data_dir: str = 'classification_processed_data',
        models_dir: str = 'saved_models',
        approach: str = 'label'
    ) -> bool:
        print("\n" + "=" * 70)
        print("LOADING DATA AND TRAINED NEURAL NETWORK MODEL")
        print("=" * 70)

        predictor = SocialWorkPredictorModels()
        data = predictor.load_preprocessed_data(data_dir=data_dir, approach=approach)

        if data is None:
            print("[ERROR] Could not load preprocessed data")
            return False

        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.feature_names = data['feature_names']

        print(f"\n[DATA] Loaded successfully")
        print(f"   Training samples: {self.X_train.shape[0]}")
        print(f"   Test samples: {self.X_test.shape[0]}")
        print(f"   Features: {len(self.feature_names)}")

        model_path = os.path.join(models_dir, 'neural_network_model.pkl')
        if os.path.exists(model_path):
            self.models['neural_network'] = joblib.load(model_path)
            print(f"[LOADED] NEURAL_NETWORK from {model_path}")
        else:
            print(f"[ERROR] neural_network model not found at {model_path}")
            return False

        return True

    def _to_2d(self, arr: np.ndarray) -> np.ndarray:
        arr = np.array(arr)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 3:
            if arr.shape[0] > 1:
                arr = arr[1]
            else:
                arr = arr[0]
        return arr

    def _feature_importance_from_shap(self, shap_values: np.ndarray) -> np.ndarray:
        sv = self._to_2d(shap_values)
        fi = np.abs(sv).mean(axis=0)
        fi = np.array(fi).reshape(-1)
        if fi.shape[0] != len(self.feature_names):
            print(
                f"[WARNING] Feature importance length {fi.shape[0]} "
                f"!= feature_names length {len(self.feature_names)}; clipping to min length"
            )
            m = min(fi.shape[0], len(self.feature_names))
            fi = fi[:m]
            self.feature_names = self.feature_names[:m]
        return fi

    def compute_shap_values(self) -> bool:
        print("\n" + "=" * 70)
        print("COMPUTING SHAP VALUES FOR NEURAL NETWORK ONLY")
        print("=" * 70)
        print("\nThis may take several minutes...")

        if 'neural_network' not in self.models:
            print("[ERROR] Neural network model not loaded")
            return False

        name = 'neural_network'
        model = self.models[name]

        print(f"\n[{name.upper()}] Computing SHAP values...")

        try:
            background = shap.sample(self.X_train, 100)
            explainer = shap.KernelExplainer(model.predict_proba, background)

            sample_size = min(100, self.X_test.shape[0])
            self.X_test_subset = self.X_test[:sample_size]
            
            shap_values_full = explainer.shap_values(self.X_test_subset)

            if isinstance(shap_values_full, list):
                shap_values = shap_values_full[1 if len(shap_values_full) > 1 else 0]
            else:
                shap_values = shap_values_full

            shap_values = np.array(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[1] if shap_values.shape[0] > 1 else shap_values[0]

            if shap_values.shape[0] != self.X_test_subset.shape[0]:
                print(f"[WARNING] SHAP values samples {shap_values.shape[0]} != X_test_subset samples {self.X_test_subset.shape[0]}")
                min_samples = min(shap_values.shape[0], self.X_test_subset.shape[0])
                shap_values = shap_values[:min_samples]
                self.X_test_subset = self.X_test_subset[:min_samples]

            if shap_values.shape[1] != self.X_test_subset.shape[1]:
                print(f"[WARNING] SHAP values features {shap_values.shape[1]} != X_test_subset features {self.X_test_subset.shape[1]}")
                min_features = min(shap_values.shape[1], self.X_test_subset.shape[1])
                shap_values = shap_values[:, :min_features]
                self.X_test_subset = self.X_test_subset[:, :min_features]
                self.feature_names = self.feature_names[:min_features]

            print(f"   [DEBUG] Final SHAP values shape: {shap_values.shape}")
            print(f"   [DEBUG] Final X_test_subset shape: {self.X_test_subset.shape}")
            print(f"   [DEBUG] Feature names count: {len(self.feature_names)}")

            feature_importance = self._feature_importance_from_shap(shap_values)

            self.shap_results[name] = {
                'explainer': explainer,
                'shap_values': shap_values,
                'feature_importance': feature_importance
            }

            print(f"   SHAP computation complete for {name.upper()}")

        except Exception as e:
            print(f"   [ERROR] Failed to compute SHAP for {name}: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    def print_feature_rankings(self) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print("SHAP FEATURE IMPORTANCE RANKINGS - NEURAL NETWORK")
        print("=" * 70)

        name = 'neural_network'
        shap_data = self.shap_results[name]
        fi = np.array(shap_data['feature_importance']).reshape(-1)

        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Importance': fi
        }).sort_values('SHAP_Importance', ascending=False)

        print(f"\n{name.replace('_', ' ').upper()} - Top 10 Features by SHAP:")
        print(f"{'Rank':<6}{'Feature':<25}{'SHAP Importance':<20}")
        print("-" * 51)

        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"{i:<6}{row['Feature']:<25}{row['SHAP_Importance']:<20.6f}")

        return importance_df

    def create_shap_visualizations(
        self,
        importance_df: pd.DataFrame,
        output_dir: str = 'classification_base_shap_analysis'
    ) -> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("\n" + "=" * 70)
        print("CREATING SHAP VISUALIZATIONS - NEURAL NETWORK")
        print("=" * 70)

        sns.set_style("whitegrid")

        top_df = importance_df.head(15)

        plt.figure(figsize=(10, 6))
        plt.barh(top_df['Feature'], top_df['SHAP_Importance'],
                 color='#8b5cf6', edgecolor='black', alpha=0.8)
        plt.xlabel('Mean |SHAP Value|', fontweight='bold')
        plt.title('NEURAL NETWORK - SHAP Feature Importance', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(output_dir, 'neural_network_shap_importance_bar.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] {out_path}")
        plt.close()

    def create_detailed_shap_plots(
        self,
        output_dir: str = 'classification_base_shap_analysis'
    ) -> None:
        print("\n" + "=" * 70)
        print("DETAILED SHAP PLOTS FOR NEURAL NETWORK")
        print("=" * 70)

        if 'neural_network' not in self.shap_results:
            print("[WARNING] Neural Network not available for detailed plots")
            return

        name = 'neural_network'
        shap_values = self.shap_results[name]['shap_values']
        explainer = self.shap_results[name]['explainer']

        print(f"\n[DEBUG] Before plotting - SHAP values shape: {shap_values.shape}")
        print(f"[DEBUG] Before plotting - X_test_subset shape: {self.X_test_subset.shape}")

        try:
            print(f"\n1. SHAP Summary Plot (Feature Impact Distribution)")
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                self.X_test_subset,
                feature_names=self.feature_names,
                show=False
            )
            out_path = os.path.join(output_dir, 'neural_network_summary_plot.png')
            plt.title('SHAP Summary Plot - NEURAL NETWORK', fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] {out_path}")
            plt.close()

        except Exception as e:
            print(f"[ERROR] Summary plot failed: {e}")
            plt.close()

        try:
            print(f"\n2. SHAP Summary Plot (Bar)")
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                self.X_test_subset,
                feature_names=self.feature_names,
                plot_type="bar",
                show=False
            )
            out_path = os.path.join(output_dir, 'neural_network_bar_plot.png')
            plt.title('SHAP Feature Importance - NEURAL NETWORK', fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] {out_path}")
            plt.close()

        except Exception as e:
            print(f"[ERROR] Bar plot failed: {e}")
            plt.close()

        try:
            print(f"\n3. SHAP Waterfall Plot (First Prediction)")
            plt.figure(figsize=(12, 8))
            
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=base_value,
                    data=self.X_test_subset[0],
                    feature_names=self.feature_names
                ),
                show=False
            )
            out_path = os.path.join(output_dir, 'neural_network_waterfall_plot.png')
            plt.title(
                'SHAP Waterfall Plot - Sample Prediction (NEURAL NETWORK)',
                fontweight='bold',
                pad=20
            )
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] {out_path}")
            plt.close()

        except Exception as e:
            print(f"[ERROR] Waterfall plot failed: {e}")
            plt.close()

        try:
            fi = np.abs(shap_values).mean(axis=0)
            top_idx = np.argsort(fi)[::-1][:3]

            print(f"\n4. SHAP Dependence Plots (Top 3 Features)")
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            for ax_idx, feat_idx in enumerate(top_idx):
                shap.dependence_plot(
                    feat_idx,
                    shap_values,
                    self.X_test_subset,
                    feature_names=self.feature_names,
                    ax=axes[ax_idx],
                    show=False
                )
                axes[ax_idx].set_title(
                    f"SHAP Dependence: {self.feature_names[feat_idx]}",
                    fontweight='bold'
                )

            plt.tight_layout()
            out_path = os.path.join(output_dir, 'neural_network_dependence_plots.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] {out_path}")
            plt.close()

        except Exception as e:
            print(f"[ERROR] Dependence plots failed: {e}")
            plt.close()

    def save_shap_results(
        self,
        importance_df: pd.DataFrame,
        output_dir: str = 'classification_base_shap_analysis'
    ) -> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        csv_path = os.path.join(output_dir, 'neural_network_shap_importance.csv')
        importance_df.to_csv(csv_path, index=False)

        print("\n" + "=" * 70)
        print("SHAP ANALYSIS COMPLETE - NEURAL NETWORK")
        print("=" * 70)
        print(f"\nSHAP importance data saved to: {output_dir}/")
        print("Files created:")
        print("   neural_network_shap_importance.csv")
        print("   neural_network_shap_importance_bar.png")
        print("   neural_network_summary_plot.png")
        print("   neural_network_bar_plot.png")
        print("   neural_network_waterfall_plot.png")
        print("   neural_network_dependence_plots.png")


def main():
    analyzer = BaseModelsSHAPAnalysis()

    print("=" * 70)
    print("BASE MODELS SHAP ANALYSIS - NEURAL NETWORK ONLY")
    print("=" * 70)

    if not analyzer.load_data_and_models():
        print("\n[ERROR] Failed to load data and neural network model")
        return

    if not analyzer.compute_shap_values():
        print("\n[ERROR] Failed to compute SHAP values for neural network")
        return

    importance_df = analyzer.print_feature_rankings()

    analyzer.create_shap_visualizations(importance_df)

    analyzer.create_detailed_shap_plots()

    analyzer.save_shap_results(importance_df)

    print("\n[COMPLETE] Neural network SHAP analysis finished successfully!")


if __name__ == "__main__":
    main()