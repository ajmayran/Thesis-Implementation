import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.neural_network import MLPClassifier

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
        self.y_train = None
        self.y_test = None

    def load_data_and_models(
        self,
        data_dir: str = 'classification_processed_data',
        models_dir: str = 'saved_models',
        approach: str = 'label'
    ) -> bool:
        print("\n" + "=" * 70)
        print("LOADING DATA FOR NEURAL NETWORK SHAP ANALYSIS")
        print("=" * 70)

        predictor = SocialWorkPredictorModels()
        data = predictor.load_preprocessed_data(data_dir=data_dir, approach=approach)

        if data is None:
            print("[ERROR] Could not load preprocessed data")
            return False

        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
        self.feature_names = data['feature_names']

        print(f"\n[DATA] Loaded successfully")
        print(f"   X_train shape: {self.X_train.shape}")
        print(f"   X_test shape:  {self.X_test.shape}")
        print(f"   y_train shape: {self.y_train.shape}")
        print(f"   y_test shape:  {self.y_test.shape}")
        print(f"   Features (len(feature_names)): {len(self.feature_names)}")
        print(f"   Feature names: {self.feature_names}")

        if self.X_train.shape[1] != len(self.feature_names):
            print("[ERROR] X_train feature count does not match feature_names length")
            return False

        if self.X_train.shape[1] <= 2:
            print("\n[ERROR] Only "
                  f"{self.X_train.shape[1]} feature(s) detected in X_train.")
            print("        That is why you only see 2 SHAP features.")
            print("        Fix classification_processed_data so it contains all features.")
            return False

        print("\n[INFO] Will train a fresh neural network in-memory for SHAP analysis.")
        return True

    def _to_2d(self, arr: np.ndarray) -> np.ndarray:
        arr = np.array(arr)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 3:
            raise ValueError("_to_2d should not receive 3D array")
        return arr

    def _feature_importance_from_shap(self, shap_values: np.ndarray) -> np.ndarray:
        sv = self._to_2d(shap_values)
        fi = np.abs(sv).mean(axis=0)
        fi = np.array(fi).reshape(-1)
        print(f"[DEBUG] feature_importance_from_shap: fi shape = {fi.shape}, "
              f"len(feature_names) = {len(self.feature_names)}")
        if fi.shape[0] != len(self.feature_names):
            print(
                f"[ERROR] Feature importance length {fi.shape[0]} "
                f"!= feature_names length {len(self.feature_names)}"
            )
        return fi

    def _train_neural_network(self):
        print("\n" + "=" * 70)
        print("TRAINING IN-MEMORY NEURAL NETWORK FOR SHAP")
        print("=" * 70)

        clf = MLPClassifier(
            hidden_layer_sizes=(50,),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42
        )

        print("[INFO] Fitting MLPClassifier on full training data...")
        clf.fit(self.X_train, self.y_train)

        train_acc = clf.score(self.X_train, self.y_train)
        test_acc = clf.score(self.X_test, self.y_test)
        print(f"[INFO] In-memory NN accuracy - Train: {train_acc:.3f}, Test: {test_acc:.3f}")

        self.models['neural_network'] = clf

    def compute_shap_values(self) -> bool:
        print("\n" + "=" * 70)
        print("COMPUTING SHAP VALUES FOR IN-MEMORY NEURAL NETWORK")
        print("=" * 70)
        print("\nThis may take several minutes...")

        if 'neural_network' not in self.models:
            self._train_neural_network()

        name = 'neural_network'
        model = self.models[name]

        print(f"\n[{name.upper()}] Computing SHAP values...")

        try:
            background = shap.sample(self.X_train, 100)
            print(f"[DEBUG] background shape: {background.shape}")

            explainer = shap.KernelExplainer(model.predict_proba, background)

            sample_size = min(100, self.X_test.shape[0])
            self.X_test_subset = self.X_test[:sample_size]
            print(f"[DEBUG] X_test_subset shape: {self.X_test_subset.shape}")

            shap_values_full = explainer.shap_values(self.X_test_subset)
            print("[DEBUG] Raw shap_values_full type:", type(shap_values_full))

            if isinstance(shap_values_full, list):
                for i, arr in enumerate(shap_values_full):
                    print(f"   shap_values_full[{i}] shape: {np.array(arr).shape}")
                if len(shap_values_full) == 2:
                    sv_pos = np.array(shap_values_full[1])
                else:
                    sv_pos = np.array(shap_values_full[0])
            else:
                sv_pos = np.array(shap_values_full)
                print(f"   shap_values_full shape: {sv_pos.shape}")

            if sv_pos.ndim == 3:
                print(f"[DEBUG] 3D array detected with shape: {sv_pos.shape}")
                if sv_pos.shape[2] == 2:
                    sv_pos = sv_pos[:, :, 1]
                    print(f"[DEBUG] Extracted positive class ([:, :, 1]). New shape: {sv_pos.shape}")
                else:
                    sv_pos = sv_pos[:, :, 0]
                    print(f"[DEBUG] Extracted first class ([:, :, 0]). New shape: {sv_pos.shape}")

            if sv_pos.ndim != 2:
                raise ValueError(f"Unexpected SHAP array shape: {sv_pos.shape}")

            print(f"[DEBUG] shap_values (positive class) shape before align: {sv_pos.shape}")

            if sv_pos.shape[0] != self.X_test_subset.shape[0]:
                print(f"[WARNING] SHAP values samples {sv_pos.shape[0]} "
                    f"!= X_test_subset samples {self.X_test_subset.shape[0]}")
                min_samples = min(sv_pos.shape[0], self.X_test_subset.shape[0])
                sv_pos = sv_pos[:min_samples]
                self.X_test_subset = self.X_test_subset[:min_samples]

            if sv_pos.shape[1] != self.X_test_subset.shape[1]:
                raise ValueError(
                    f"Feature mismatch: shap_values has {sv_pos.shape[1]} features "
                    f"but X_test_subset has {self.X_test_subset.shape[1]}"
                )

            if sv_pos.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Feature mismatch: shap_values has {sv_pos.shape[1]} features "
                    f"but feature_names has {len(self.feature_names)}"
                )

            shap_values = sv_pos

            print(f"   [DEBUG] Final SHAP values shape: {shap_values.shape}")
            print(f"   [DEBUG] Final X_test_subset shape: {self.X_test_subset.shape}")
            print(f"   [DEBUG] Feature names count: {len(self.feature_names)}")
            print(f"   [DEBUG] Feature names: {self.feature_names}")

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

        print(f"[DEBUG] print_feature_rankings: fi shape = {fi.shape}")
        print(f"[DEBUG] feature_names length = {len(self.feature_names)}")

        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Importance': fi
        }).sort_values('SHAP_Importance', ascending=False)

        print(f"\n{name.replace('_', ' ').upper()} - All Features by SHAP:")
        print(f"{'Rank':<6}{'Feature':<25}{'SHAP Importance':<20}")
        print("-" * 51)

        for i, (_, row) in enumerate(importance_df.iterrows(), 1):
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

        top_df = importance_df.head(len(importance_df))

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
            sorted_idx = np.argsort(fi)[::-1]

            print(f"\n4. SHAP Dependence Plots (ALL Features, ordered by importance)")
            n_features = len(sorted_idx)
            cols = 3
            rows = int(np.ceil(n_features / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
            axes = np.array(axes).reshape(-1)

            for ax_idx, feat_idx in enumerate(sorted_idx):
                ax = axes[ax_idx]
                shap.dependence_plot(
                    feat_idx,
                    shap_values,
                    self.X_test_subset,
                    feature_names=self.feature_names,
                    ax=ax,
                    show=False
                )
                ax.set_title(
                    f"SHAP Dependence: {self.feature_names[feat_idx]}",
                    fontweight='bold'
                )

            for extra_ax in axes[len(sorted_idx):]:
                extra_ax.axis('off')

            plt.tight_layout()
            out_path = os.path.join(output_dir, 'neural_network_dependence_plots_all_features.png')
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

        full_csv_path = os.path.join(output_dir, 'neural_network_shap_full_importance.csv')
        importance_df.to_csv(full_csv_path, index=False)

        print("\n" + "=" * 70)
        print("SHAP ANALYSIS COMPLETE - NEURAL NETWORK")
        print("=" * 70)
        print(f"\nSHAP importance data saved to: {output_dir}/")
        print("   neural_network_shap_importance.csv")
        print("   neural_network_shap_full_importance.csv")
        print("   neural_network_shap_importance_bar.png")
        print("   neural_network_summary_plot.png")
        print("   neural_network_bar_plot.png")
        print("   neural_network_waterfall_plot.png")
        print("   neural_network_dependence_plots_all_features.png")


def main():
    analyzer = BaseModelsSHAPAnalysis()

    print("=" * 70)
    print("BASE MODELS SHAP ANALYSIS - IN-MEMORY NEURAL NETWORK")
    print("=" * 70)

    if not analyzer.load_data_and_models():
        print("\n[ERROR] Failed to load data")
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