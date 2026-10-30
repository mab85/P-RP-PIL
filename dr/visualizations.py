
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
import pandas as pd
import os
import time
import gc
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_diabetic_retinopathy_binary
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

OUTPUT_DIR = 'visualization_output_dr'


class DRVisualizer:
    def __init__(self, model, class_names, output_dir=OUTPUT_DIR):
        self.model = model
        self.class_names = class_names
        self.output_dir = output_dir
        self.baseline_results = {}
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('default')
        sns.set_style("whitegrid")

    # ------------------------------------------------------------------
    def set_prp(self, y_pred, acc, time_s):
        self.baseline_results['P-RP-PIL'] = dict(
            training_time=time_s, accuracy=acc, predictions=y_pred)
        print(f"   P-RP-PIL recorded: Acc={acc:.4f}, Time={time_s:.2f}s")

    # ------------------------------------------------------------------
    def build_dl(self, name):
        nc = len(self.class_names)
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
        else:  # Custom-CNN
            m = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(128, 128, 3)),
                Conv2D(32, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(), Dense(512, activation='relu'), Dropout(0.5),
                Dense(nc, activation='softmax')])
            m.compile(optimizer=Adam(1e-3),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
            return m

        base.trainable = False
        inputs = tf.keras.Input(shape=(128, 128, 3))
        x = base(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(h, activation='relu')(x)
        x = Dropout(0.3)(x)
        out = Dense(nc, activation='softmax')(x)
        m = Model(inputs, out)
        m.compile(optimizer=Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        return m

    # ------------------------------------------------------------------
    def train_baselines(self, X_tr, y_tr, X_va, y_va, X_te, y_te):
        nc = len(self.class_names)
        y_tr_c = to_categorical(y_tr, num_classes=nc)
        y_va_c = to_categorical(y_va, num_classes=nc)
        cw_arr = compute_class_weight('balanced',
                                      classes=np.arange(nc), y=y_tr)
        cw = {i: float(w) for i, w in enumerate(cw_arr)}
        cb = [EarlyStopping(patience=5, restore_best_weights=True),
              ReduceLROnPlateau(factor=0.5, patience=3)]

        for name in ['ResNet-50', 'DenseNet-121', 'MobileNetV2',
                     'EfficientNet-B0', 'Custom-CNN']:
            try:
                print(f"  Training {name}...")
                model = self.build_dl(name)
                # EfficientNet-B0 expects [0, 255]
                scale = 255.0 if name == 'EfficientNet-B0' else 1.0
                t0 = time.time()
                model.fit(X_tr * scale, y_tr_c,
                          validation_data=(X_va * scale, y_va_c),
                          epochs=30, batch_size=32,
                          class_weight=cw, callbacks=cb, verbose=0)
                t = time.time() - t0
                pred = np.argmax(
                    model.predict(X_te * scale, verbose=0), axis=1)
                acc = accuracy_score(y_te, pred)
                self.baseline_results[name] = dict(
                    training_time=t, accuracy=acc, predictions=pred)
                print(f"    ✓ {name}: {acc:.4f} in {t:.1f}s")
            except Exception as e:
                print(f"    ✗ {name} failed: {e}")

    # ------------------------------------------------------------------
    def _save(self, fig, name):
        p = os.path.join(self.output_dir, name)
        fig.tight_layout()
        fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"   Saved: {p}")

    # ------------------------------------------------------------------
    def fig_confusion(self, y_true, y_pred):
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        acc = np.trace(cm) / np.sum(cm)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names, ax=ax)
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('True', fontweight='bold')
        ax.set_title(f'P-RP-PIL Confusion Matrix (DR) — Acc {acc:.3f}',
                     fontweight='bold')
        self._save(fig, 'confusion_matrix_dr.png')

    # ------------------------------------------------------------------
    def fig_performance_metrics(self, y_true, y_pred):
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(self.class_names)); w = 0.25
        precs, recs, f1s = [], [], []
        for c in range(len(self.class_names)):
            yt = (y_true == c).astype(int)
            yp = (y_pred == c).astype(int)
            precs.append(precision_score(yt, yp, zero_division=0))
            recs.append(recall_score(yt, yp, zero_division=0))
            f1s.append(f1_score(yt, yp, zero_division=0))
        ax.bar(x - w, precs, w, label='Precision', color='#2E86AB')
        ax.bar(x,     recs,  w, label='Recall',    color='#A23B72')
        ax.bar(x + w, f1s,   w, label='F1-Score',  color='#F18F01')
        for i, (p, r, f) in enumerate(zip(precs, recs, f1s)):
            ax.text(i - w, p + 0.02, f'{p:.2f}', ha='center', fontsize=8)
            ax.text(i,     r + 0.02, f'{r:.2f}', ha='center', fontsize=8)
            ax.text(i + w, f + 0.02, f'{f:.2f}', ha='center', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(self.class_names)
        ax.set_ylim(0, 1.1); ax.legend()
        ax.set_title('Per-Class Performance Metrics (DR)',
                     fontweight='bold')
        self._save(fig, 'performance_metrics_dr.png')

    # ------------------------------------------------------------------
    def fig_training_insights(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        names = list(self.baseline_results.keys())
        times = [self.baseline_results[n]['training_time'] for n in names]
        accs  = [self.baseline_results[n]['accuracy'] for n in names]
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        sizes = [300 + 200 * (a - min(accs)) / (max(accs) - min(accs) + 1e-6)
                 for a in accs]
        ax.scatter(times, accs, s=sizes, c=colors,
                   edgecolors='black', alpha=0.8)
        for n, t, a in zip(names, times, accs):
            ax.annotate(f'{n}\n({a:.3f}, {t:.1f}s)', (t, a),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="white", alpha=0.7))
        ax.set_xlabel('Training Time (seconds)', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('Accuracy vs Training Time (DR) — All Models',
                     fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save(fig, 'Figure_S8_DR_Train.png')

    # ------------------------------------------------------------------
    def fig_per_class_heatmap(self, y_true, y_pred):
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        rows = []
        for i in range(len(self.class_names)):
            tp = cm[i, i]; fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            s = tn / (tn + fp) if (tn + fp) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            rows.append([p, r, s, f])
        im = ax.imshow(np.array(rows), cmap='YlOrRd', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Score')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Precision', 'Recall', 'Specificity', 'F1'])
        ax.set_yticks(range(len(self.class_names)))
        ax.set_yticklabels(self.class_names)
        for i in range(len(rows)):
            for j in range(4):
                ax.text(j, i, f'{rows[i][j]:.2f}',
                        ha='center', va='center',
                        color='white' if rows[i][j] > 0.5 else 'black',
                        fontweight='bold')
        ax.set_title('Per-Class Metrics Heatmap (DR)', fontweight='bold')
        self._save(fig, 'per_class_metrics_dr.png')

    
    def fig_attention(self, image):
        try:
            from prp_pil import PRP_PIL
            if not isinstance(self.model, PRP_PIL):
                raise TypeError("model is not PRP_PIL")
            if not self.model.use_attention:
                raise RuntimeError("attention disabled in model")

            # Use the first scale only for the spatial map
            ps = self.model.patch_sizes[0]
            st = self.model.strides[0]
            R = self.model.R_list[0]

            H, W = image.shape[:2]
            weights = np.zeros((H, W), dtype=np.float32)
            counts = np.zeros((H, W), dtype=np.float32)
            raw_weights = []

            for i in range(0, H - ps + 1, st):
                for j in range(0, W - ps + 1, st):
                    patch = image[i:i+ps, j:j+ps, :].flatten()
                    z = np.tanh(patch @ R)
                    v = float(z.var())
                    raw_weights.append(v)
                    weights[i:i+ps, j:j+ps] += v
                    counts[i:i+ps, j:j+ps] += 1

            weights = weights / (counts + 1e-8)
            w = weights - weights.min()
            w = w / (w.max() + 1e-8)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(image, alpha=0.55)
            im = ax.imshow(w, cmap='jet', alpha=0.45)
            ax.set_title(
                f'Attention Heatmap (DR) — patch {ps}×{ps}, stride {st}',
                fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, label='Attention weight')
            self._save(fig, 'Figure_S6_DR_Attn.png')
        except Exception as e:
            print(f"  attention heatmap skipped: {e}")

    # ------------------------------------------------------------------
    def fig_architecture(self):
        steps = ['Input (128×128×3)',
                 'Patch Extraction (6×6, stride 3)',
                 'Random Projection (dim=256)',
                 'Attention Pooling (variance-based)',
                 'Hidden Layer (2048, tanh)',
                 'Pseudo-Inverse Solve (λ=1e-5)',
                 'Classification (2 classes)']
        fig, ax = plt.subplots(figsize=(8, 6))
        y = np.arange(len(steps))
        ax.barh(y, [1] * len(steps),
                color=plt.cm.Blues(np.linspace(0.3, 1, len(steps))))
        ax.set_yticks(y); ax.set_yticklabels(steps, fontsize=9)
        ax.set_xticks([]); ax.set_xlim(0, 1.2)
        for i in range(len(steps)):
            ax.text(1.02, i, f'Step {i+1}', va='center', fontsize=8)
        ax.set_title('P-RP-PIL Architecture (DR)', fontweight='bold')
        self._save(fig, 'model_architecture_dr.png')

    # ------------------------------------------------------------------
    def fig_comparison_conf(self, y_test):
        names = list(self.baseline_results.keys())
        n = len(names)
        cols = min(3, n); rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.atleast_1d(axes).ravel()
        i = -1
        for i, name in enumerate(names):
            cm = confusion_matrix(
                y_test, self.baseline_results[name]['predictions'])
            acc = np.trace(cm) / np.sum(cm)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names, ax=axes[i])
            axes[i].set_title(f'{name}\n(Acc {acc:.3f})',
                              fontweight='bold')
            axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('Actual')
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        self._save(fig, 'comparison_confusion_matrices_dr.png')

    # ------------------------------------------------------------------
    def fig_roc_pr(self, y_true, y_proba):
        fig, ax = plt.subplots(figsize=(8, 6))
        proba_pos = y_proba[:, 1] if y_proba.shape[1] == 2 \
                    else y_proba[:, -1]
        fpr, tpr, _ = roc_curve(y_true, proba_pos)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC (AUC = {auc_val:.3f})')
        ax.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--',
                label='Random')
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curve (DR)', fontweight='bold')
        ax.legend(loc='lower right'); ax.grid(alpha=0.3)
        self._save(fig, 'roc_curve_dr.png')

        fig, ax = plt.subplots(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, proba_pos)
        ap = np.mean(precision)
        ax.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR (AP = {ap:.3f})')
        ax.set_xlabel('Recall', fontweight='bold')
        ax.set_ylabel('Precision', fontweight='bold')
        ax.set_title('Precision-Recall Curve (DR)', fontweight='bold')
        ax.legend(loc='lower left'); ax.grid(alpha=0.3)
        self._save(fig, 'precision_recall_curve_dr.png')


# ======================================================================
def main():
    print("=" * 70)
    print("GENERATING DR VISUALIZATIONS")
    print("=" * 70)

    X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
        load_diabetic_retinopathy_binary(
            img_size=(128, 128), max_samples_per_class=None,
            use_retino=False)
    if X_tr is None:
        print(" Failed to load DR dataset."); return
    print(f"Train {X_tr.shape[0]}  Val {X_va.shape[0]}  "
          f"Test {X_te.shape[0]}")

    print("\n Training P-RP-PIL on DR ...")
    model = PRP_PIL(
        patch_size=6, stride=3, projection_dim=256, n_hidden=2048,
        activation='tanh', use_advanced_features=True,
        use_attention=True, use_color_hist=False, use_gradients=True,
        regularization=1e-5, random_state=42, batch_size=200)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    t_prp = time.time() - t0
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"   P-RP-PIL: Acc={acc:.4f}, Time={t_prp:.2f}s")

    viz = DRVisualizer(model, list(le.classes_))
    viz.set_prp(y_pred, acc, t_prp)
    viz.train_baselines(X_tr, y_tr, X_va, y_va, X_te, y_te)

    viz.fig_confusion(y_te, y_pred)
    viz.fig_performance_metrics(y_te, y_pred)
    viz.fig_training_insights()
    viz.fig_per_class_heatmap(y_te, y_pred)
    viz.fig_attention(X_te[0])
    viz.fig_feature_importance()
    viz.fig_architecture()
    viz.fig_comparison_conf(y_te)
    try:
        y_proba = model.predict_proba(X_te)
        viz.fig_roc_pr(y_te, y_proba)
    except Exception as e:
        print(f"ROC/PR skipped: {e}")

    print(f"\n All figures saved to {viz.output_dir}/")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()