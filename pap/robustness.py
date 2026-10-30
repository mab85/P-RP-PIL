
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import data_loader
from prp_pil import PRP_PIL

import tensorflow as tf
from tensorflow.keras.applications import (ResNet50, DenseNet121,
                                           MobileNetV2, EfficientNetB0)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout,
                                     Conv2D, MaxPooling2D, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import argparse


# ======================================================================
# 1. Model builders
# ======================================================================

def build_transfer(name, num_classes, fine_tune_layers=20):
    """
    Build a transfer-learning model with partial fine-tuning.

    fine_tune_layers : number of top layers of the backbone to unfreeze.
                       Set to 0 to keep the backbone fully frozen.
    """
    if name == 'ResNet-50':
        base = ResNet50(weights='imagenet', include_top=False,
                        input_shape=(128, 128, 3))
        head_dim = 512
    elif name == 'DenseNet-121':
        base = DenseNet121(weights='imagenet', include_top=False,
                           input_shape=(128, 128, 3))
        head_dim = 512
    elif name == 'MobileNetV2':
        base = MobileNetV2(weights='imagenet', include_top=False,
                           input_shape=(128, 128, 3))
        head_dim = 256
    elif name == 'EfficientNet-B0':
        base = EfficientNetB0(weights='imagenet', include_top=False,
                              input_shape=(128, 128, 3))
        head_dim = 256
    else:
        raise ValueError(f"Unknown model: {name}")

    # --- Sanity check: verify that ImageNet weights actually loaded ---
    if len(base.weights) == 0:
        raise RuntimeError(f"{name}: no weights found; ImageNet load failed.")
    total_params = sum(int(np.prod(w.shape)) for w in base.weights)
    if total_params < 1_000_000:
        raise RuntimeError(
            f"{name}: weight count too small ({total_params}). "
            f"ImageNet load likely failed.")

    # --- Partial fine-tuning: unfreeze top layers ---
    base.trainable = False
    if fine_tune_layers > 0:
        for layer in base.layers[-fine_tune_layers:]:
            # Never unfreeze BatchNorm running statistics on small datasets
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    # --- Classification head ---
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(head_dim, activation='relu')(x)
    x = Dropout(0.5)(x)                 # stronger dropout for small datasets
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=5e-4),   # lower LR because we are fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


def build_custom_cnn(num_classes):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])
    m.compile(optimizer=Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m


# ======================================================================
# 2. Per-model input preprocessing
# ======================================================================

def prepare_input(X, model_name):
    """
    EfficientNet-B0 has a built-in Rescaling(1/127.5) layer that expects
    inputs in [0, 255]. All other models expect inputs in [0, 1].
    """
    if model_name == 'EfficientNet-B0':
        return X * 255.0
    return X


# ======================================================================
# 3. Dataset loaders
# ======================================================================

def load_dataset(dataset, img_size=(128, 128), max_per_class=800):
    if dataset == 'papilledema':
        return data_loader.load_papilledema_dataset(
            img_size=img_size, max_samples_per_class=max_per_class)

    if hasattr(data_loader, 'load_dr_dataset'):
        return data_loader.load_dr_dataset(
            img_size=img_size, max_samples_per_class=max_per_class)
    elif hasattr(data_loader, 'combine_datasets_smart'):
        return data_loader.combine_datasets_smart(
            img_size=img_size, max_samples_per_class=max_per_class)
    elif hasattr(data_loader, 'load_diabetic_retinopathy_binary'):
        return data_loader.load_diabetic_retinopathy_binary(
            img_size=img_size, max_samples_per_class=max_per_class,
            use_retino=False)
    else:
        raise AttributeError("No DR loader found in data_loader.py.")


# ======================================================================
# 4. Perturbations
# ======================================================================

def add_noise(X, sigma=0.1):
    return np.clip(X + np.random.normal(0, sigma, X.shape), 0, 1)

def add_brightness(X, factor):
    return np.clip(X * factor, 0, 1)


# ======================================================================
# 5. Main
# ======================================================================

def main(dataset='papilledema'):
    print(f"\n=== Robustness test — {dataset} ===")
    num_classes = 3 if dataset == 'papilledema' else 2

    # ---- Load dataset ----
    X, y = load_dataset(dataset)
    if X is None:
        print(" Failed to load dataset.")
        return

    # ---- Split ----
    if dataset == 'papilledema':
        X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
            data_loader.preprocess_medical_data(
                X, y, test_size=0.15, val_size=0.15, random_state=42)
    else:
        X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
            data_loader.preprocess_medical_data_binary(
                X, y, test_size=0.15, val_size=0.15, random_state=42)
    del X, y

    # ---- Perturbed test sets ----
    X_te_noise  = add_noise(X_te, sigma=0.1)
    X_te_bright = add_brightness(X_te, factor=1.2)

    rows = []

    # ==================== P-RP-PIL ====================
    print("\nTraining P-RP-PIL ...")
    if dataset == 'papilledema':
        model = PRP_PIL(
            patch_size=8, stride=2, projection_dim=256, n_hidden=2048,
            activation='tanh', use_advanced_features=True, use_attention=True,
            use_gradients=True, regularization=1e-3,
            random_state=42, batch_size=200)
    else:
        model = PRP_PIL(
            patch_size=6, stride=1, projection_dim=512, n_hidden=2048,
            activation='tanh', use_advanced_features=True, use_attention=True,
            use_gradients=True, regularization=1e-5,
            random_state=42, batch_size=200)

    model.fit(X_tr, y_tr)
    a_clean  = accuracy_score(y_te, model.predict(X_te))
    a_noise  = accuracy_score(y_te, model.predict(X_te_noise))
    a_bright = accuracy_score(y_te, model.predict(X_te_bright))
    rows.append({
        'Model': 'P-RP-PIL',
        'Clean':       f"{a_clean:.4f}",
        'Noise':       f"{a_noise:.4f}",
        'Noise Drop':  f"{(a_clean - a_noise) * 100:.2f}",
        'Bright':      f"{a_bright:.4f}",
        'Bright Drop': f"{(a_clean - a_bright) * 100:.2f}",
    })
    print(f"P-RP-PIL        : clean {a_clean:.4f} | noise {a_noise:.4f} "
          f"| bright {a_bright:.4f}")

    # ==================== DL baselines ====================
    y_tr_c = to_categorical(y_tr, num_classes=num_classes)
    y_va_c = to_categorical(y_va, num_classes=num_classes)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-6, verbose=0),
    ]

    # Majority-class baseline for the sanity check
    maj_class = np.bincount(y_te, minlength=num_classes).argmax()
    maj_acc = np.mean(y_te == maj_class)
    print(f"Majority-class accuracy on test set: {maj_acc:.4f}")

    for name in ['ResNet-50', 'DenseNet-121', 'MobileNetV2',
                 'EfficientNet-B0', 'Custom-CNN']:
        print(f"\nTraining {name} ...")
        try:
            m = build_custom_cnn(num_classes) if name == 'Custom-CNN' \
                else build_transfer(name, num_classes, fine_tune_layers=20)

            # Per-model input scaling
            X_tr_m        = prepare_input(X_tr, name)
            X_va_m        = prepare_input(X_va, name)
            X_te_m        = prepare_input(X_te, name)
            X_te_noise_m  = prepare_input(X_te_noise, name)
            X_te_bright_m = prepare_input(X_te_bright, name)

            m.fit(X_tr_m, y_tr_c,
                  validation_data=(X_va_m, y_va_c),
                  epochs=40, batch_size=32,
                  callbacks=callbacks, verbose=0)

            a_clean  = accuracy_score(y_te, np.argmax(m.predict(X_te_m, verbose=0), axis=1))
            a_noise  = accuracy_score(y_te, np.argmax(m.predict(X_te_noise_m, verbose=0), axis=1))
            a_bright = accuracy_score(y_te, np.argmax(m.predict(X_te_bright_m, verbose=0), axis=1))

            # --- Sanity check: detect majority-class collapse ---
            if a_clean < maj_acc + 0.05:
                print(f"  ⚠️ {name}: clean acc {a_clean:.4f} is close to "
                      f"majority-class accuracy {maj_acc:.4f}. "
                      f"Possible training failure.")

            rows.append({
                'Model': name,
                'Clean':       f"{a_clean:.4f}",
                'Noise':       f"{a_noise:.4f}",
                'Noise Drop':  f"{(a_clean - a_noise) * 100:.2f}",
                'Bright':      f"{a_bright:.4f}",
                'Bright Drop': f"{(a_clean - a_bright) * 100:.2f}",
            })
            print(f"  ✓ {name}: clean {a_clean:.4f} | noise {a_noise:.4f} "
                  f"| bright {a_bright:.4f}")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            rows.append({'Model': name, 'Clean': '—', 'Noise': '—',
                         'Noise Drop': '—', 'Bright': '—', 'Bright Drop': '—'})

    # ---- Save CSV ----
    df = pd.DataFrame(rows)
    out_csv = f'Table_S3_Robustness_{dataset}.csv'
    df.to_csv(out_csv, index=False)
    print(f"\n Saved: {out_csv}")
    print(df.to_string(index=False))

    # ---- LaTeX-ready body ----
    print("\n--- LaTeX body ---")
    for _, r in df.iterrows():
        print(f"{r['Model']} & {r['Clean']} & {r['Noise']} & {r['Noise Drop']} "
              f"& {r['Bright']} & {r['Bright Drop']} \\\\")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='papilledema',
                        choices=['papilledema', 'dr'])
    args = parser.parse_args()
    np.random.seed(42)
    tf.random.set_seed(42)
    main(args.dataset)