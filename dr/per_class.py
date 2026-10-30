
import os
import gc
import time
import random
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')


os.environ.setdefault('TF_DETERMINISTIC_OPS', '0')
os.environ.setdefault('CUDNN_DETERMINISTIC', '0')

os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_auto_jit=2')

warnings.filterwarnings('ignore')


import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import (ResNet50, DenseNet121,
                                           MobileNetV2, EfficientNetB0)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                     Dropout, Conv2D, MaxPooling2D,
                                     Flatten, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Local modules (after TF so they can use GPU config if needed)
from data_loader import load_diabetic_retinopathy_binary
from prp_pil import PRP_PIL


# ======================================================================
# 3) GPU CONFIGURATION HELPER  (call from main(), not at import time)
# ======================================================================
def configure_gpu(enable_mixed_precision: bool = False,
                  memory_limit_mb: int | None = None):
    """
    Configure GPU memory growth and (optionally) mixed precision.

    Must be called AFTER `import tensorflow as tf`.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("  No GPU detected — running on CPU.")
        return

    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
            if memory_limit_mb is not None:
                tf.config.set_logical_device_configuration(
                    g,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=memory_limit_mb)]
                )
        logical = tf.config.list_logical_devices('GPU')
        print(f" GPU(s) available: {[d.name for d in logical]}")
    except RuntimeError as e:
        # GPU already initialized — memory growth can't be changed now.
        print(f"  GPU already initialized: {e}")

    if enable_mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print(" Mixed precision (mixed_float16) enabled.")
        except Exception as e:
            print(f"  Could not enable mixed precision: {e}")


# ======================================================================
# 4) EVALUATOR
# ======================================================================
class ModelEvaluator:
    def __init__(self):
        self.results = {}
        self.class_names = ['No_DR', 'DR']
        self.class_descriptions = {
            'No_DR': 'No Diabetic Retinopathy',
            'DR':    'Diabetic Retinopathy',
        }

    # ------------------------------------------------------------------
    def load_data(self):
        print(" Loading DR dataset (author-provided split)...")
        X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
            load_diabetic_retinopathy_binary(
                img_size=(128, 128), max_samples_per_class=None,
                use_retino=False)
        if X_tr is None:
            raise ValueError("Could not load DR dataset!")

        print(f"  Train: {X_tr.shape[0]}  "
              f"Val: {X_va.shape[0]}  Test: {X_te.shape[0]}")
        print(f"  Test class distribution: "
              f"No_DR={(y_te == 0).sum()}  DR={(y_te == 1).sum()}")

        return X_tr, X_va, X_te, y_tr, y_va, y_te, le

    # ------------------------------------------------------------------
    def build_prp_pil(self):
        """Production DR configuration."""
        return PRP_PIL(
            patch_size=6, stride=3, projection_dim=256,
            n_hidden=2048, activation='tanh',
            use_advanced_features=True, use_attention=True,
            use_color_hist=False, use_gradients=True,
            regularization=1e-5, random_state=42, batch_size=200)

    # ------------------------------------------------------------------
    def _build_transfer(self, name, num_classes=2):
        if name == 'ResNet-50':
            base = ResNet50(weights='imagenet', include_top=False,
                            input_shape=(128, 128, 3)); h = 512
        elif name == 'DenseNet-121':
            base = DenseNet121(weights='imagenet', include_top=False,
                               input_shape=(128, 128, 3)); h = 512
        elif name == 'MobileNetV2':
            base = MobileNetV2(weights='imagenet', include_top=False,
                               input_shape=(128, 128, 3)); h = 256
        elif name == 'EfficientNet-B0':
            base = EfficientNetB0(weights='imagenet', include_top=False,
                                  input_shape=(128, 128, 3)); h = 256
        else:
            raise ValueError(name)

        base.trainable = False
        inputs = Input(shape=(128, 128, 3))
        x = base(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(h, activation='relu')(x)
        x = Dropout(0.3)(x)
        # Cast back to float32 for stable softmax under mixed_float16
        out = Dense(num_classes, activation='softmax',
                    dtype='float32')(x)
        m = Model(inputs, out)
        m.compile(optimizer=Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        return m

    def build_custom_cnn(self, num_classes=2):
        m = tf.keras.Sequential([
            Input(shape=(128, 128, 3)),
            Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax', dtype='float32'),
        ])
        m.compile(optimizer=Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        return m

    # ------------------------------------------------------------------
    def calculate_per_class_metrics(self, y_true, y_pred, model_name):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        class_metrics = {}
        for class_idx, class_name in enumerate(self.class_names):
            tp = int(((y_true == class_idx) & (y_pred == class_idx)).sum())
            fp = int(((y_true != class_idx) & (y_pred == class_idx)).sum())
            fn = int(((y_true == class_idx) & (y_pred != class_idx)).sum())
            tn = int(((y_true != class_idx) & (y_pred != class_idx)).sum())
            support = int((y_true == class_idx).sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)
                    if (prec + rec) > 0 else 0.0)
            bal_acc = (rec + spec) / 2

            class_metrics[class_name] = {
                'Support': support,
                'Precision': prec,
                'Recall (Sensitivity)': rec,
                'Specificity': spec,
                'F1-Score': f1,
                'Balanced Accuracy': bal_acc,
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            }

        return {
            'model_name': model_name,
            'class_metrics': class_metrics,
            'overall_accuracy': accuracy_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred,
                                                 labels=[0, 1]),
            'y_true': y_true,
            'y_pred': y_pred,
        }

    # ------------------------------------------------------------------
    def train_and_evaluate_model(self, model_name, model_builder,
                                 X_tr, X_va, X_te, y_tr, y_va, y_te,
                                 is_prp_pil=False, extra_input_scale=False):
        print(f"\n🔧 Training {model_name}...")

        if is_prp_pil:
            t0 = time.time()
            model = model_builder()
            model.fit(X_tr, y_tr)
            training_time = time.time() - t0
            y_pred = model.predict(X_te)
        else:
            num_classes = len(self.class_names)
            y_tr_c = to_categorical(y_tr, num_classes=num_classes)
            y_va_c = to_categorical(y_va, num_classes=num_classes)

            cw_arr = compute_class_weight(
                'balanced', classes=np.arange(num_classes), y=y_tr)
            cw = {i: float(w) for i, w in enumerate(cw_arr)}

            X_tr_m = X_tr * 255.0 if extra_input_scale else X_tr
            X_va_m = X_va * 255.0 if extra_input_scale else X_va
            X_te_m = X_te * 255.0 if extra_input_scale else X_te

            model = model_builder()
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5,
                              restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, verbose=0),
            ]
            t0 = time.time()
            model.fit(X_tr_m, y_tr_c,
                      validation_data=(X_va_m, y_va_c),
                      epochs=30, batch_size=32,
                      class_weight=cw,
                      callbacks=callbacks, verbose=0)
            training_time = time.time() - t0
            y_pred = np.argmax(model.predict(X_te_m, verbose=0), axis=1)

        results = self.calculate_per_class_metrics(y_te, y_pred, model_name)
        results['training_time'] = training_time
        print(f"  ✓ {model_name}: Acc={results['overall_accuracy']:.4f}, "
              f"Time={training_time:.1f}s")

        # Free GPU memory between models
        try:
            del model
        except Exception:
            pass
        tf.keras.backend.clear_session()
        gc.collect()

        return results

    # ------------------------------------------------------------------
    def generate_report(self):
        print("\n" + "=" * 70)
        print("Per-Class Performance — all models on the DR test set")
        print("=" * 70)
        X_tr, X_va, X_te, y_tr, y_va, y_te, le = self.load_data()

        models_to_eval = [
            ('P-RP-PIL',        lambda: self.build_prp_pil(),
             True,  False),
            ('ResNet-50',       lambda: self._build_transfer('ResNet-50'),
             False, False),
            ('DenseNet-121',    lambda: self._build_transfer('DenseNet-121'),
             False, False),
            ('MobileNetV2',     lambda: self._build_transfer('MobileNetV2'),
             False, False),
            ('EfficientNet-B0', lambda: self._build_transfer('EfficientNet-B0'),
             False, True),       # EfficientNet-B0 gets [0, 255]
            ('Custom-CNN',      lambda: self.build_custom_cnn(),
             False, False),
        ]

        for name, builder, is_prp, extra_scale in models_to_eval:
            try:
                tf.keras.utils.set_random_seed(42)
                np.random.seed(42)
                random.seed(42)
                r = self.train_and_evaluate_model(
                    name, builder, X_tr, X_va, X_te,
                    y_tr, y_va, y_te,
                    is_prp_pil=is_prp, extra_input_scale=extra_scale)
                self.results[name] = r
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")

        self.create_comparison_table()
        self.create_visualizations()
        del X_tr, X_va, X_te, y_tr, y_va, y_te
        gc.collect()
        return self.results

    # ------------------------------------------------------------------
    def create_comparison_table(self):
        rows = []
        for model_name, r in self.results.items():
            for cls in self.class_names:
                cm = r['class_metrics'][cls]
                rows.append({
                    'Model': model_name,
                    'Class': cls,
                    'Precision': cm['Precision'],
                    'Recall':    cm['Recall (Sensitivity)'],
                    'Specificity': cm['Specificity'],
                    'F1-Score':  cm['F1-Score'],
                    'Balanced Acc': cm['Balanced Accuracy'],
                    'Support':   cm['Support'],
                })
        df = pd.DataFrame(rows)
        df.to_csv('per_class_performance_comparison.csv', index=False)
        print("\n Saved: per_class_performance_comparison.csv")

        rows2 = []
        for model_name, r in self.results.items():
            rows2.append({
                'Model': model_name,
                'Accuracy (%)': f"{r['overall_accuracy']*100:.2f}",
                'Training Time (s)': f"{r['training_time']:.1f}",
                'Macro Precision':
                    np.mean([r['class_metrics'][c]['Precision']
                             for c in self.class_names]),
                'Macro Recall':
                    np.mean([r['class_metrics'][c]['Recall (Sensitivity)']
                             for c in self.class_names]),
                'Macro Specificity':
                    np.mean([r['class_metrics'][c]['Specificity']
                             for c in self.class_names]),
                'Macro F1':
                    np.mean([r['class_metrics'][c]['F1-Score']
                             for c in self.class_names]),
            })
        df2 = pd.DataFrame(rows2)
        df2.to_csv('dr_model_comparison_results.csv', index=False)
        print(" Saved: dr_model_comparison_results.csv")
        return df

    # ------------------------------------------------------------------
    def create_visualizations(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette('husl')

        # Individual confusion matrices — grid
        n = len(self.results)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.atleast_1d(axes).ravel()
        i = -1
        for i, (name, r) in enumerate(self.results.items()):
            cm = r['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names, ax=axes[i])
            axes[i].set_title(f"{name}\n(Acc {r['overall_accuracy']:.3f})")
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.savefig('comparison_confusion_matrices_dr.png', dpi=300,
                    bbox_inches='tight')
        plt.close()
        print(" Saved: comparison_confusion_matrices_dr.png")

        # Per-class grouped bar chart
        metrics = ['Precision', 'Recall (Sensitivity)', 'Specificity',
                   'F1-Score', 'Balanced Accuracy']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        for idx, metric in enumerate(metrics):
            rows_data = []
            for model_name, r in self.results.items():
                for cls in self.class_names:
                    val = r['class_metrics'][cls].get(metric, 0.0)
                    rows_data.append({
                        'Model': model_name, 'Class': cls, 'Value': val,
                    })
            pdf = pd.DataFrame(rows_data)
            sns.barplot(data=pdf, x='Model', y='Value', hue='Class',
                        ax=axes[idx], alpha=0.85)
            axes[idx].set_title(metric, fontweight='bold')
            axes[idx].set_ylim(0, 1.05)
            axes[idx].tick_params(axis='x', rotation=45)

        models = list(self.results.keys())
        accs = [self.results[m]['overall_accuracy'] for m in models]
        axes[-1].bar(models, accs,
                     color=sns.color_palette('viridis', len(models)))
        axes[-1].set_title('Overall Accuracy', fontweight='bold')
        axes[-1].set_ylim(0, 1.05)
        axes[-1].tick_params(axis='x', rotation=45)
        for b, a in zip(axes[-1].patches, accs):
            axes[-1].text(b.get_x() + b.get_width() / 2,
                          b.get_height() + 0.01,
                          f'{a:.3f}', ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig('per_class_performance_visualization.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(" Saved: per_class_performance_visualization.png")

    # ------------------------------------------------------------------
    def write_markdown_report(self):
        with open('per_class_performance_report.md', 'w') as f:
            f.write("# Per-Class Performance Report (DR)\n\n")
            f.write("## Models evaluated\n\n")
            f.write("- P-RP-PIL\n- ResNet-50\n- DenseNet-121\n")
            f.write("- MobileNetV2\n- EfficientNet-B0\n- Custom-CNN\n\n")
            f.write("## Overall accuracy\n\n")
            f.write("| Model | Accuracy | Training Time (s) |\n")
            f.write("|-------|----------|-------------------|\n")
            for name, r in self.results.items():
                f.write(f"| {name} | {r['overall_accuracy']:.4f} | "
                        f"{r['training_time']:.1f} |\n")
            f.write("\n## Per-class detail\n\n")
            for name, r in self.results.items():
                f.write(f"### {name}\n\n")
                f.write("| Class | Precision | Recall | Specificity | "
                        "F1 | Support |\n")
                f.write("|-------|-----------|--------|-------------|"
                        "----|---------|\n")
                for cls in self.class_names:
                    cm = r['class_metrics'][cls]
                    f.write(f"| {cls} | {cm['Precision']:.4f} | "
                            f"{cm['Recall (Sensitivity)']:.4f} | "
                            f"{cm['Specificity']:.4f} | "
                            f"{cm['F1-Score']:.4f} | "
                            f"{cm['Support']} |\n")
                f.write("\n")
        print(" Saved: per_class_performance_report.md")


# ======================================================================
# 5) MAIN
# ======================================================================
def main():
    # Seeds
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)

    # Configure GPU now that TF is imported
    configure_gpu(enable_mixed_precision=False)   # set True on Ampere+ if wanted

    print(" Per-Class Performance — DR dataset")
    evaluator = ModelEvaluator()
    evaluator.generate_report()
    evaluator.write_markdown_report()
    print("\n Done.")


if __name__ == '__main__':
    main()