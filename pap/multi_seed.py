# Usage:
#   python multi_seed_eval.py --dataset papilledema
#   python multi_seed_eval.py --dataset dr
# ----------------------------------------------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import argparse
import data_loader
from prp_pil import PRP_PIL
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)


# ----------------------------------------------------------------------
# Dataset-specific model hyperparameters
# ----------------------------------------------------------------------
def get_model_kwargs(dataset):
    if dataset == 'papilledema':
        return dict(
            patch_size=8, stride=2, projection_dim=256, n_hidden=2048,
            activation='tanh', use_advanced_features=True, use_attention=True,
            use_color_hist=False, use_gradients=True, regularization=1e-3,
            batch_size=200)
    else:
        return dict(
            patch_size=6, stride=1, projection_dim=512, n_hidden=2048,
            activation='tanh', use_advanced_features=True, use_attention=True,
            use_color_hist=False, use_gradients=True, regularization=1e-5,
            batch_size=200)


# ----------------------------------------------------------------------
# Per-seed experiment
# ----------------------------------------------------------------------
def run_single_seed(X, y, seed, model_kwargs):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.15, random_state=seed, stratify=y_enc)

    # Model uses seed for random projection + hidden weights
    kwargs = dict(model_kwargs)
    kwargs['random_state'] = seed

    model = PRP_PIL(**kwargs)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)

    # Basic metrics
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)

    # AUC (macro OvR for multi-class, single for binary)
    try:
        if n_classes == 2:
            proba = model.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, proba)
        else:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_te, classes=np.arange(n_classes))
            proba = model.predict_proba(X_te)
            auc = roc_auc_score(y_bin, proba, average='macro', multi_class='ovr')
    except Exception:
        auc = float('nan')

    # Macro-averaged specificity
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_te, y_pred, labels=np.arange(n_classes))
    specs = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    spec = float(np.mean(specs))

    return dict(seed=seed, accuracy=acc, precision=prec, recall=rec,
                specificity=spec, f1=f1, auc=auc)


# ----------------------------------------------------------------------
# Confidence interval helper (t-distribution, small sample)
# ----------------------------------------------------------------------
def ci95(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return float('nan'), float('nan')
    mean = values.mean()
    sem = values.std(ddof=1) / np.sqrt(n)
    # t critical for 95% with n-1 df (approximation for n=10: 2.262)
    t_crit = {9: 2.262, 8: 2.306, 7: 2.365, 6: 2.447, 5: 2.571,
              4: 2.776, 3: 3.182}.get(n - 1, 1.96)
    margin = t_crit * sem
    return mean - margin, mean + margin


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(dataset='papilledema', n_seeds=10):
    print(f"\n=== Multi-seed evaluation — {dataset} — {n_seeds} seeds ===\n")

    # Load dataset once
    if dataset == 'papilledema':
        X, y = data_loader.load_papilledema_dataset(
            img_size=(128, 128), max_samples_per_class=800)
    else:
        if hasattr(data_loader, 'load_dr_dataset'):
            X, y = data_loader.load_dr_dataset(
                img_size=(128, 128), max_samples_per_class=800)
        elif hasattr(data_loader, 'combine_datasets_smart'):
            X, y = data_loader.combine_datasets_smart(
                img_size=(128, 128), max_samples_per_class=800)
        else:
            X, y = data_loader.load_diabetic_retinopathy_binary(
                img_size=(128, 128), max_samples_per_class=800,
                use_retino=False)

    if X is None:
        print(" Failed to load dataset."); return

    model_kwargs = get_model_kwargs(dataset)

    rows = []
    for seed in range(n_seeds):
        print(f"  Seed {seed:2d} ...", end='', flush=True)
        try:
            res = run_single_seed(X, y, seed, model_kwargs)
            rows.append(res)
            print(f"  acc={res['accuracy']*100:.2f}%  f1={res['f1']:.4f}  auc={res['auc']:.4f}")
        except Exception as e:
            print(f"  ✗ failed: {e}")

    df = pd.DataFrame(rows)
    csv_raw = f'Table_S2_Multi_Seed_raw_{dataset}.csv'
    df.to_csv(csv_raw, index=False)

    # Summary with mean ± SD and 95% CI
    summary_rows = []
    for metric in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']:
        vals = df[metric].values.astype(float)
        mean = np.nanmean(vals) * 100 if metric == 'accuracy' else np.nanmean(vals)
        std = np.nanstd(vals, ddof=1) * 100 if metric == 'accuracy' else np.nanstd(vals, ddof=1)
        lo, hi = ci95(vals)
        if metric == 'accuracy':
            lo, hi = lo * 100, hi * 100
        summary_rows.append({
            'Metric': metric.capitalize(),
            'Mean': f"{mean:.4f}",
            'SD': f"{std:.4f}",
            '95% CI lower': f"{lo:.4f}",
            '95% CI upper': f"{hi:.4f}",
        })

    summary = pd.DataFrame(summary_rows)
    csv_summary = f'Table_S2_Multi_Seed_raw_{dataset}.csv'
    summary.to_csv(csv_summary, index=False)

    print(f"\n Raw results:     {csv_raw}")
    print(f" Summary:         {csv_summary}")
    print("\nSummary table:")
    print(summary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='papilledema',
                        choices=['papilledema', 'dr'])
    parser.add_argument('--seeds', type=int, default=10)
    args = parser.parse_args()
    np.random.seed(42)
    main(args.dataset, args.seeds)