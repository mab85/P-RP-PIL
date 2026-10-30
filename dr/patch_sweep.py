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
def train_one(X_tr, y_tr, X_te, y_te, patch_size, stride, proj_dim=256,
              regularization=1e-3):
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
        regularization=regularization,
        random_state=42,
        batch_size=200)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    t = time.time() - t0
    acc = accuracy_score(y_te, model.predict(X_te))
    return acc, t


# ----------------------------------------------------------------------
def main(dataset='papilledema'):
    print(f"\n=== Patch-size and stride sweep — {dataset} ===")

    # ---- Load the split ----
    if dataset == 'papilledema':
        X, y = data_loader.load_papilledema_dataset(
            img_size=(128, 128), max_samples_per_class=None)
        if X is None:
            print(" Failed to load Papilledema dataset."); return
        X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
            data_loader.preprocess_medical_data(
                X, y, test_size=0.15, val_size=0.15, random_state=42)
        del X, y
        reg = 1e-3
    else:
        X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
            data_loader.load_diabetic_retinopathy_binary(
                img_size=(128, 128), max_samples_per_class=None,
                use_retino=False)
        if X_tr is None:
            print(" Failed to load DR dataset."); return
        reg = 1e-5

    print(f"Train: {X_tr.shape[0]}  Val: {X_va.shape[0]}  "
          f"Test: {X_te.shape[0]}")

    # ---- Sweep grid ----
    patch_sizes = [4, 6, 8, 12, 16]
    strides     = [1, 2, 3, 4]

    rows = []
    for ps in patch_sizes:
        for st in strides:
            if st >= ps:
                continue
            print(f"  patch={ps:2d}  stride={st}  ...", end='', flush=True)
            try:
                acc, t = train_one(X_tr, y_tr, X_te, y_te, ps, st,
                                   proj_dim=256, regularization=reg)
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