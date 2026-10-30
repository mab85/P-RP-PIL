--------------------------------------------------------

import numpy as np
import time
import os
import gc
import pandas as pd

import tensorflow as tf
tf.keras.utils.set_random_seed(42)

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from data_loader import load_diabetic_retinopathy_binary
from prp_pil import PRP_PIL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.applications import (ResNet50, DenseNet121,
                                           MobileNetV2, EfficientNetB0)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                     Dropout, Conv2D, MaxPooling2D, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight


# ----------------------------------------------------------------------
# Dataset check
# ----------------------------------------------------------------------
def setup_dataset():
    data_dir = 'data'
    required = [os.path.join(data_dir, "Diagnosis of Diabetic Retinopathy"),
                os.path.join(data_dir, "Retino")]
    existing = [d for d in required if os.path.exists(d)]
    if not existing:
        print("Error: DR dataset not found!")
        return False
    print(f"Found datasets: {[os.path.basename(d) for d in existing]}")
    return True


# ----------------------------------------------------------------------
# Per-model input scaling
# ----------------------------------------------------------------------
def prepare_for_model(X, model_name):
    """
    EfficientNet-B0 has an internal Rescaling(1/127.5) layer that
    expects inputs in [0, 255]. All other models expect [0, 1].
    """
    if model_name == 'EfficientNet-B0':
        return X * 255.0
    return X


# ----------------------------------------------------------------------
# DL model builders
# ----------------------------------------------------------------------
def build_transfer(name, num_classes):
    if name == 'ResNet-50':
        base = ResNet50(weights='imagenet', include_top=False,
                        input_shape=(128, 128, 3)); hidden = 512
    elif name == 'DenseNet-121':
        base = DenseNet121(weights='imagenet', include_top=False,
                           input_shape=(128, 128, 3)); hidden = 512
    elif name == 'MobileNetV2':
        base = MobileNetV2(weights='imagenet', include_top=False,
                           input_shape=(128, 128, 3)); hidden = 256
    elif name == 'EfficientNet-B0':
        base = EfficientNetB0(weights='imagenet', include_top=False,
                              input_shape=(128, 128, 3)); hidden = 256
    else:
        raise ValueError(name)

    base.trainable = False
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(hidden, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    m = Model(inputs, out)
    m.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m


def build_custom_cnn(num_classes):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
        Flatten(), Dense(512, activation='relu'), Dropout(0.5),
        Dense(num_classes, activation='softmax')])
    m.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    return dict(acc=accuracy_score(y_true, y_pred),
                precision=precision, recall=recall,
                specificity=specificity, f1=f1)


# ----------------------------------------------------------------------
# DL training — clean version (no try/except, no device switching)
# ----------------------------------------------------------------------
def train_dl(name, builder, X_tr, y_tr, X_va, y_va, X_te, y_te, n_classes):
    X_tr_in = prepare_for_model(X_tr, name)
    X_va_in = prepare_for_model(X_va, name)
    X_te_in = prepare_for_model(X_te, name)

    y_tr_c = to_categorical(y_tr, num_classes=n_classes)
    y_va_c = to_categorical(y_va, num_classes=n_classes)

    cw_arr = compute_class_weight('balanced',
                                  classes=np.arange(n_classes), y=y_tr)
    class_weight = {i: float(w) for i, w in enumerate(cw_arr)}

    cb = [EarlyStopping(patience=5, restore_best_weights=True),
          ReduceLROnPlateau(factor=0.5, patience=3)]

    model = builder(n_classes)
    t0 = time.time()
    model.fit(X_tr_in, y_tr_c,
              validation_data=(X_va_in, y_va_c),
              epochs=30, batch_size=32,
              class_weight=class_weight,
              callbacks=cb, verbose=0)
    t = time.time() - t0

    pred = np.argmax(model.predict(X_te_in, verbose=0), axis=1)

    # Debug — printed for every model so mismatches are visible immediately
    cm = confusion_matrix(y_te, pred, labels=[0, 1])
    print(f"    {name} confusion matrix (rows=true, cols=pred):")
    print(f"      {cm.tolist()}")

    m = compute_metrics(y_te, pred)
    m['time'] = t
    return m, pred


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if not setup_dataset():
        return
    print("="*90)
    print("DR COMPARISON TABLE (all baselines + P-RP-PIL)")
    print("="*90)

    print("\n Loading DR dataset (author-provided splits)...")
    X_tr, X_va, X_te, y_tr, y_va, y_te, le = load_diabetic_retinopathy_binary(
        img_size=(128, 128), max_samples_per_class=None, use_retino=False)
    if X_tr is None:
        return

    print(f" Loaded:")
    print(f"   Train {X_tr.shape[0]}  Val {X_va.shape[0]}  "
          f"Test {X_te.shape[0]}")
    print(f"   Classes: {list(le.classes_)}")

    n_classes = 2
    results = []

    # ---------------- P-RP-PIL (best sweep config) ----------------
    print("\n1. Training P-RP-PIL...")
    try:
        prp = PRP_PIL(
            patch_size=6, stride=3, projection_dim=256,
            n_hidden=2048, activation='tanh',
            use_advanced_features=True, use_attention=True,
            use_color_hist=False, use_gradients=True,
            regularization=1e-5, random_state=42, batch_size=200)
        t0 = time.time()
        prp.fit(X_tr, y_tr)
        t = time.time() - t0
        pred = prp.predict(X_te)
        m_prp = compute_metrics(y_te, pred)
        m_prp['time'] = t
        cm_prp = confusion_matrix(y_te, pred, labels=[0, 1])
        print(f"    P-RP-PIL confusion matrix (rows=true, cols=pred):")
        print(f"      {cm_prp.tolist()}")
        results.append(('P-RP-PIL', m_prp))
        print(f"  ✓ P-RP-PIL: Acc={m_prp['acc']:.4f}, Time={t:.2f}s")
    except Exception as e:
        print(f"  ✗ P-RP-PIL failed: {e}")

    # ---------------- DL baselines ----------------
    dl_list = [
        ('ResNet-50',       lambda nc: build_transfer('ResNet-50', nc)),
        ('DenseNet-121',    lambda nc: build_transfer('DenseNet-121', nc)),
        ('MobileNetV2',     lambda nc: build_transfer('MobileNetV2', nc)),
        ('EfficientNet-B0', lambda nc: build_transfer('EfficientNet-B0', nc)),
        ('Custom-CNN',      build_custom_cnn),
    ]
    for name, builder in dl_list:
        print(f"\nTraining {name}...")
        try:
            tf.keras.utils.set_random_seed(42)
            np.random.seed(42)
            import random; random.seed(42)
            m, _ = train_dl(name, builder, X_tr, y_tr, X_va, y_va,
                            X_te, y_te, n_classes)
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            continue                          # do NOT leak `m` across iterations
        results.append((name, m))
        print(f"  ✓ {name}: Acc={m['acc']:.4f}, Time={m['time']:.1f}s")

    # ---------------- Save table ----------------
    df = pd.DataFrame([{
        'Model': r[0],
        'Accuracy (%)': f"{r[1]['acc']*100:.2f}",
        'Training Time (s)': f"{r[1]['time']:.1f}",
        'Precision': f"{r[1]['precision']:.4f}",
        'Recall (Sensitivity)': f"{r[1]['recall']:.4f}",
        'Specificity': f"{r[1]['specificity']:.4f}",
        'F1-Score': f"{r[1]['f1']:.4f}"
    } for r in results])
    print("\n" + "="*90)
    print(df.to_string(index=False))
    df.to_csv('dr_comparison_results.csv', index=False)
    print("\n Saved: dr_comparison_results.csv")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()