
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
import time, gc, os, warnings
warnings.filterwarnings('ignore')

import data_loader
from prp_pil import PRP_PIL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.applications import (ResNet50, DenseNet121,
                                           MobileNetV2, EfficientNetB0)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                     Dropout, Conv2D, MaxPooling2D, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight


OUTPUT_DIR = 'visualization_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================================================================
class Evaluator:
    def __init__(self, class_names):
        self.class_names = class_names
        self.results = {}

    # ------------------------------------------------------------------
    def _build_transfer(self, name, num_classes):
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
        inputs = tf.keras.Input(shape=(128, 128, 3))
        x = base(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(h, activation='relu')(x)
        x = Dropout(0.3)(x)
        out = Dense(num_classes, activation='softmax')(x)
        m = Model(inputs, out)
        m.compile(optimizer=Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        return m

    def _build_custom_cnn(self, num_classes):
        m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 128, 3)),
            Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
            Flatten(), Dense(512, activation='relu'), Dropout(0.5),
            Dense(num_classes, activation='softmax')])
        m.compile(optimizer=Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        return m

    # ------------------------------------------------------------------
    def _per_class(self, y_true, y_pred):
        out = {}
        for idx, name in enumerate(self.class_names):
            tp = int(((y_true == idx) & (y_pred == idx)).sum())
            fp = int(((y_true != idx) & (y_pred == idx)).sum())
            fn = int(((y_true == idx) & (y_pred != idx)).sum())
            tn = int(((y_true != idx) & (y_pred != idx)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            s = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            out[name] = dict(Precision=p, Recall=r, Specificity=s, F1=f,
                             Balanced=(r + s) / 2,
                             Support=int((y_true == idx).sum()),
                             TP=tp, FP=fp, FN=fn, TN=tn)
        return out

    # ------------------------------------------------------------------
    def run(self, X_tr, X_va, X_te, y_tr, y_va, y_te):
        n_classes = len(self.class_names)

        # ---- P-RP-PIL ----
        print("\n[P-RP-PIL]")
        prp = PRP_PIL(
            patch_size=8, stride=2, projection_dim=256,
            n_hidden=2048, activation='tanh',
            use_advanced_features=True, use_attention=True,
            use_color_hist=False, use_gradients=True,
            regularization=1e-3, random_state=42, batch_size=200)
        t0 = time.time()
        prp.fit(X_tr, y_tr)
        t_prp = time.time() - t0
        pred_prp = prp.predict(X_te)
        self.results['P-RP-PIL'] = dict(
            pred=pred_prp, time=t_prp,
            acc=accuracy_score(y_te, pred_prp),
            per_class=self._per_class(y_te, pred_prp))
        print(f"  acc={self.results['P-RP-PIL']['acc']:.4f}  time={t_prp:.1f}s")

        # ---- DL baselines ----
        y_tr_c = to_categorical(y_tr, num_classes=n_classes)
        y_va_c = to_categorical(y_va, num_classes=n_classes)
        cw_arr = compute_class_weight('balanced',
                                      classes=np.arange(n_classes), y=y_tr)
        cw = {i: float(w) for i, w in enumerate(cw_arr)}
        cb = [EarlyStopping(monitor='val_loss', patience=5,
                            restore_best_weights=True, verbose=0),
              ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=3, verbose=0)]

        for name in ['ResNet-50', 'DenseNet-121', 'MobileNetV2',
                     'EfficientNet-B0', 'Custom-CNN']:
            print(f"\n[{name}]")
            try:
                model = (self._build_custom_cnn(n_classes)
                         if name == 'Custom-CNN'
                         else self._build_transfer(name, n_classes))
                scale = 255.0 if name == 'EfficientNet-B0' else 1.0
                t0 = time.time()
                model.fit(X_tr * scale, y_tr_c,
                          validation_data=(X_va * scale, y_va_c),
                          epochs=30, batch_size=32, class_weight=cw,
                          callbacks=cb, verbose=0)
                t = time.time() - t0
                pred = np.argmax(model.predict(X_te * scale, verbose=0), axis=1)
                self.results[name] = dict(
                    pred=pred, time=t,
                    acc=accuracy_score(y_te, pred),
                    per_class=self._per_class(y_te, pred))
                print(f"  acc={self.results[name]['acc']:.4f}  time={t:.1f}s")
            except Exception as e:
                print(f"  ✗ failed: {e}")

        return self.results

    # ------------------------------------------------------------------
    def save_csvs(self):
        # Per-class long table
        rows = []
        for model, r in self.results.items():
            for cls in self.class_names:
                m = r['per_class'][cls]
                rows.append({
                    'Model': model, 'Class': cls,
                    'Precision': m['Precision'],
                    'Recall (Sensitivity)': m['Recall'],
                    'Specificity': m['Specificity'],
                    'F1-Score': m['F1'],
                    'Support': m['Support'],
                })
        pd.DataFrame(rows).to_csv('papilledema_detailed_metrics.csv',
                                  index=False)

        # Summary wide table
        rows2 = []
        for model, r in self.results.items():
            pc = r['per_class']
            rows2.append({
                'Model': model,
                'Accuracy (%)': f"{r['acc']*100:.2f}",
                'Training Time (s)': f"{r['time']:.1f}",
                'Macro Precision': np.mean([pc[c]['Precision']
                                            for c in self.class_names]),
                'Macro Recall (Sensitivity)': np.mean([pc[c]['Recall']
                                                       for c in self.class_names]),
                'Macro Specificity': np.mean([pc[c]['Specificity']
                                              for c in self.class_names]),
                'Macro F1-Score': np.mean([pc[c]['F1']
                                           for c in self.class_names]),
            })
        pd.DataFrame(rows2).to_csv('papilledema_model_comparison_results.csv',
                                   index=False)
        print("\n Saved: papilledema_detailed_metrics.csv")
        print(" Saved: papilledema_model_comparison_results.csv")

    # ------------------------------------------------------------------
    def fig_per_class(self):
        """Fig 2 — grouped bar chart, all 6 models × 3 classes."""
        metrics = [('Precision', 'Precision Comparison'),
                   ('Recall', 'Recall (Sensitivity) Comparison'),
                   ('Specificity', 'Specificity Comparison'),
                   ('F1', 'F1-Score Comparison'),
                   ('Balanced', 'Balanced Accuracy Comparison')]

        model_order = ['P-RP-PIL', 'ResNet-50', 'DenseNet-121',
                       'MobileNetV2', 'EfficientNet-B0', 'Custom-CNN']
        model_order = [m for m in model_order if m in self.results]

        fig, axes = plt.subplots(2, 3, figsize=(19, 11))
        axes = axes.ravel()
        palette = {'Normal': '#3FA7B6', 'Papilledema': '#A83B7A',
                   'Pseudopapilledema': '#F09724'}

        for idx, (key, title) in enumerate(metrics):
            ax = axes[idx]
            x = np.arange(len(model_order))
            width = 0.26
            for k, cls in enumerate(self.class_names):
                vals = [self.results[m]['per_class'][cls][key]
                        for m in model_order]
                bars = ax.bar(x + (k - 1) * width, vals, width,
                              label=cls, color=palette.get(cls, None),
                              alpha=0.88)
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2,
                            b.get_height() + 0.01, f'{v:.3f}',
                            ha='center', va='bottom', fontsize=7,
                            rotation=90)
            ax.set_xticks(x)
            ax.set_xticklabels(model_order, rotation=30, ha='right',
                               fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(key)
            ax.set_title(title, fontweight='bold')
            if idx == 0:
                ax.legend(title='Class', fontsize=8, loc='lower left')

        # Last panel: overall accuracy
        ax = axes[-1]
        accs = [self.results[m]['acc'] for m in model_order]
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(model_order)))
        bars = ax.bar(model_order, accs, color=colors)
        for b, a in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f'{a:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Accuracy')
        ax.set_title('Overall Performance Comparison', fontweight='bold')
        ax.tick_params(axis='x', rotation=30)

        fig.suptitle('Per-Class Performance for Papilledema Classification',
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        fig.savefig('papilledema_per_class_performance.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(" Saved: papilledema_per_class_performance.png")

    # ------------------------------------------------------------------
    def fig_confusion(self, y_te):
        """Fig 3 — confusion matrices, all 6 models."""
        model_order = ['P-RP-PIL', 'ResNet-50', 'DenseNet-121',
                       'MobileNetV2', 'EfficientNet-B0', 'Custom-CNN']
        model_order = [m for m in model_order if m in self.results]

        n = len(model_order)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = np.atleast_1d(axes).ravel()
        for i, name in enumerate(model_order):
            cm = confusion_matrix(y_te, self.results[name]['pred'],
                                  labels=list(range(len(self.class_names))))
            acc = np.trace(cm) / cm.sum()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names, ax=axes[i],
                        cbar_kws={'shrink': 0.8})
            axes[i].set_title(f'{name}\n(Acc: {acc:.3f})',
                              fontweight='bold', fontsize=11)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].tick_params(axis='y', rotation=0)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        p = os.path.join(OUTPUT_DIR, 'comparison_confusion_matrices.png')
        fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f" Saved: {p}")


# ======================================================================
def main():
    print("=" * 70)
    print("PAPILLEDEMA PER-CLASS FIGURES — production config")
    print("=" * 70)

    X, y = data_loader.load_papilledema_dataset(
        img_size=(128, 128), max_samples_per_class=None)
    if X is None:
        print(" Could not load Papilledema dataset."); return

    X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
        data_loader.preprocess_medical_data(
            X, y, test_size=0.15, val_size=0.15, random_state=42)
    del X, y; gc.collect()

    print(f"Train {X_tr.shape[0]} | Val {X_va.shape[0]} | "
          f"Test {X_te.shape[0]}")
    print(f"Classes: {list(le.classes_)}")

    evaluator = Evaluator(list(le.classes_))
    evaluator.run(X_tr, X_va, X_te, y_tr, y_va, y_te)
    evaluator.save_csvs()
    evaluator.fig_per_class()
    evaluator.fig_confusion(y_te)

    print("\n Done.")


if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    main()