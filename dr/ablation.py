# Usage:
#   python r3c6_ablation.py --dataset papilledema
#   python r3c6_ablation.py --dataset dr
# ----------------------------------------------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import time
import argparse
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
import data_loader
from prp_pil import PRP_PIL


# ----------------------------------------------------------------------
# Per-dataset production configuration
# ----------------------------------------------------------------------
def production_config(dataset):
    if dataset == 'papilledema':
        return dict(
            # multi-scale (production: ON)
            use_multi_scale=True,
            patch_sizes=[4, 8, 12], strides=[2, 4, 6],
            projection_dims=[128, 256, 384],
            # explicit single-scale fallback (used only when the
            # no-multi-scale row is evaluated)
            patch_size=8, stride=2, projection_dim=256,
            # feature toggles (production: all ON except color_hist)
            use_attention=True, use_random_projection=True, use_pca=True,
            use_gabor=True, use_rff=True,
            use_binary_hashing=True, use_graph_laplacian=True,
            use_advanced_features=True, use_gradients=True,
            use_color_hist=False,
            # shared
            n_hidden=2048, activation='tanh',
            regularization=1e-3, random_state=42, batch_size=200,
        )
    else:  # dr
        return dict(
            # single-scale production config (matches baseline.py)
            patch_size=6, stride=3, projection_dim=256,
            use_multi_scale=False,
            # explicit multi-scale fallback (used only when the
            # with-multi-scale row is evaluated)
            patch_sizes=[4, 6, 8], strides=[1, 2, 3],
            projection_dims=[128, 256, 384],
            # feature toggles (production: only 3 ON)
            use_attention=True, use_random_projection=True, use_pca=True,
            use_gabor=False, use_rff=False,
            use_binary_hashing=False, use_graph_laplacian=False,
            use_advanced_features=True, use_gradients=True,
            use_color_hist=False,
            # shared
            n_hidden=2048, activation='tanh',
            regularization=1e-5, random_state=42, batch_size=200,
        )


# ----------------------------------------------------------------------
# Load + split (70/15/15) per dataset
# ----------------------------------------------------------------------
def load_and_split(dataset):
    if dataset == 'papilledema':
        X, y = data_loader.load_papilledema_dataset(
            img_size=(128, 128), max_samples_per_class=None)
        return data_loader.preprocess_medical_data(
            X, y, test_size=0.15, val_size=0.15, random_state=42)
    else:
        return data_loader.load_diabetic_retinopathy_binary(
            img_size=(128, 128), max_samples_per_class=None,
            use_retino=False)


# ----------------------------------------------------------------------
# Row definitions
# ----------------------------------------------------------------------
PAP_ROWS = [
    ('Full P-RP-PIL',                     None, None),
    ('w/o multi-scale patches',          'use_multi_scale', False),
    ('w/o attention weighting',          'use_attention', False),
    ('w/o random projection',            'use_random_projection', False),
    ('w/o PCA',                          'use_pca', False),
    ('w/o Gabor features',               'use_gabor', False),
    ('w/o Random Fourier Features',      'use_rff', False),
    ('w/o binary hashing',               'use_binary_hashing', False),
    ('w/o graph-Laplacian',              'use_graph_laplacian', False),
    ('w/o PIL (linear SVM on features)', 'PIL', 'linear'),
    ('w/o PIL (RBF SVM on features)',    'PIL', 'rbf'),
]

DR_ROWS = [
    ('Full P-RP-PIL',                     None, None),
    ('with multi-scale patches',         'use_multi_scale', True),
    ('w/o attention weighting',          'use_attention', False),
    ('w/o random projection',            'use_random_projection', False),
    ('w/o PCA',                          'use_pca', False),
    ('with Gabor features',              'use_gabor', True),
    ('with Random Fourier Features',     'use_rff', True),
    ('with binary hashing',              'use_binary_hashing', True),
    ('with graph-Laplacian',             'use_graph_laplacian', True),
    ('w/o PIL (linear SVM on features)', 'PIL', 'linear'),
    ('w/o PIL (RBF SVM on features)',    'PIL', 'rbf'),
]


def build_model(cfg):
    keys = ['patch_size', 'stride', 'projection_dim', 'n_hidden',
            'activation', 'use_multi_scale', 'patch_sizes', 'strides',
            'projection_dims', 'use_advanced_features', 'use_attention',
            'use_random_projection', 'use_pca', 'use_gabor', 'use_rff',
            'use_binary_hashing', 'use_graph_laplacian', 'use_color_hist',
            'use_gradients', 'regularization', 'graph_lambda',
            'rff_dim', 'rff_gamma', 'hashing_dim', 'random_state',
            'batch_size']
    return PRP_PIL(**{k: cfg[k] for k in keys if k in cfg})


# ----------------------------------------------------------------------
def main(dataset):
    print(f"\n=== R3C6 component-wise ablation — {dataset} ===\n")
    X_tr, X_va, X_te, y_tr, y_va, y_te, le = load_and_split(dataset)
    print(f"Train {len(X_tr)} | Val {len(X_va)} | Test {len(X_te)}")

    base_cfg = production_config(dataset)
    rows = PAP_ROWS if dataset == 'papilledema' else DR_ROWS

    results, base_acc = [], None
    for label, comp, val in rows:
        cfg = dict(base_cfg)
        print(f"  {label} ...", end='', flush=True)

        if comp == 'PIL':
            ext = build_model(cfg)
            ext.fit(X_tr, y_tr)
            H_tr = ext._transform(X_tr).astype(np.float64)
            H_te = ext._transform(X_te).astype(np.float64)

            if val == 'linear':
                clf = LinearSVC(C=1.0, max_iter=10000,
                                random_state=42)
            else:
                clf = SVC(kernel='rbf', C=1.0, gamma='scale',
                          random_state=42)
            clf.fit(H_tr, y_tr)
            acc = accuracy_score(y_te, clf.predict(H_te))
            t_str = '—'
        else:
            if comp is not None:
                cfg[comp] = val
            model = build_model(cfg)
            t0 = time.time()
            model.fit(X_tr, y_tr)
            t_str = round(time.time() - t0, 1)
            acc = accuracy_score(y_te, model.predict(X_te))

        if base_acc is None:
            base_acc = acc
        delta = (acc - base_acc) * 100
        results.append({
            'Configuration': label,
            'Accuracy (%)':  round(acc * 100, 2),
            'Delta (pp)':    round(delta, 2),
            'Time (s)':      t_str,
        })
        print(f"  acc={acc*100:.2f}%   Δ={delta:+.2f} pp")

    df = pd.DataFrame(results)
    out = 'Table_Ablation_Pap.csv' if dataset == 'papilledema' else 'Table_S6_DR_Ablation.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='papilledema',
                    choices=['papilledema', 'dr'])
    args = ap.parse_args()
    np.random.seed(42)
    main(args.dataset)