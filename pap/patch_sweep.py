# Usage:
#   python patch_stride_sweep.py --dataset papilledema
#   python patch_stride_sweep.py --dataset dr
# ----------------------------------------------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import time
import argparse
import data_loader
from prp_pil import PRP_PIL
from sklearn.metrics import accuracy_score


# ----------------------------------------------------------------------
# Train and evaluate one model for a given (patch_size, stride)
# ----------------------------------------------------------------------
def train_one(X_tr, y_tr, X_te, y_te, patch_size, stride, proj_dim=256):
    """Train a P-RP-PIL model with a single patch size and stride."""
    model = PRP_PIL(
        patch_size=patch_size,
        stride=stride,
        projection_dim=proj_dim,
        n_hidden=2048,
        activation='tanh',
        use_advanced_features=True,
        use_attention=True,
        use_color_hist=False,
        use_gradients=True,
        regularization=1e-3,
        random_state=42,
        batch_size=200)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    t = time.time() - t0
    acc = accuracy_score(y_te, model.predict(X_te))
    return acc, t


# ----------------------------------------------------------------------
# Load dataset for the chosen task
# ----------------------------------------------------------------------
def load_dataset(dataset, img_size=(128, 128), max_per_class=800):
    if dataset == 'papilledema':
        return data_loader.load_papilledema_dataset(
            img_size=img_size, max_samples_per_class=max_per_class)

    # DR — pick the loader that exists in this data_loader.py
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
    raise AttributeError("No DR loader found in data_loader.py.")


# ----------------------------------------------------------------------
# Main — sweep patch size and stride
# ----------------------------------------------------------------------
def main(dataset='papilledema'):
    print(f"\n=== Patch-size and stride sweep — {dataset} ===")

    # ---- Load ----
    X, y = load_dataset(dataset)
    if X is None:
        print(" Failed to load dataset."); return

    # ---- Split (same 70/15/15 as the manuscript) ----
    if dataset == 'papilledema':
        X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
            data_loader.preprocess_medical_data(
                X, y, test_size=0.15, val_size=0.15, random_state=42)
    else:
        X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
            data_loader.preprocess_medical_data_binary(
                X, y, test_size=0.15, val_size=0.15, random_state=42)

    print(f"Train: {X_tr.shape[0]}  Val: {X_va.shape[0]}  Test: {X_te.shape[0]}")

    # ---- Sweep grid ----
    patch_sizes = [4, 8, 12, 16]
    strides     = [1, 2, 4]

    rows = []
    for ps in patch_sizes:
        for st in strides:
            if st >= ps:
                # A stride equal to or larger than the patch size is invalid
                continue
            print(f"  patch={ps:2d}  stride={st}  ...", end='', flush=True)
            try:
                acc, t = train_one(X_tr, y_tr, X_te, y_te, ps, st)
                rows.append({
                    'Patch size': ps,
                    'Stride': st,
                    'Accuracy (%)': round(acc * 100, 2),
                    'Time (s)': round(t, 1),
                })
                print(f"  acc={acc*100:.2f}%  time={t:.1f}s")
            except Exception as e:
                print(f"  ✗ failed: {e}")
                rows.append({
                    'Patch size': ps, 'Stride': st,
                    'Accuracy (%)': None, 'Time (s)': None,
                })

    # ---- Save ----
    df = pd.DataFrame(rows)
    out_csv = 'Table_S4_Pap_Sweep.csv' if dataset == 'papilledema' else 'Table_S5_DR_Sweep.csv'
    df.to_csv(out_csv, index=False)
    print(f"\n Saved: {out_csv}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='papilledema',
                        choices=['papilledema', 'dr'])
    args = parser.parse_args()
    np.random.seed(42)
    main(args.dataset)