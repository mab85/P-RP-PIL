# Usage:
#   python supplementary_L_sweep.py --dataset dr
#   python supplementary_L_sweep.py --dataset papilledema
# ----------------------------------------------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import pandas as pd
import time

import data_loader
from prp_pil import PRP_PIL
from sklearn.metrics import accuracy_score


# ----------------------------------------------------------------------
# Fixed sweep design
# ----------------------------------------------------------------------
PATCH_SIZE = 16                              # raw patch dim D = 16*16*3 = 768
STRIDE     = 4
L_VALUES   = [128, 256, 384, 512, 768]
D_RAW      = PATCH_SIZE * PATCH_SIZE * 3     # = 768


# ----------------------------------------------------------------------
# Loader helpers
# ----------------------------------------------------------------------
def _call_first_available(names, **kwargs):
    """Call the first function that exists in data_loader from `names`."""
    for name in names:
        fn = getattr(data_loader, name, None)
        if callable(fn):
            print(f"  using data_loader.{name}")
            return fn(**kwargs)
    available = [x for x in dir(data_loader)
                 if not x.startswith('_') and
                 ('load' in x or 'pap' in x or 'dr' in x)]
    raise AttributeError(
        f"None of {names} found in data_loader.py. Available: {available}")


def _first_present(names):
    """Return the first attribute from `names` that exists in data_loader."""
    for name in names:
        fn = getattr(data_loader, name, None)
        if callable(fn):
            return fn
    return None


def load_dataset(dataset):
    """
    Returns (X_tr, X_va, X_te, y_tr, y_va, y_te, le).
    """
    # ------------------------------------------------------------------
    # Papilledema: loader returns (X, y); we split it here.
    # ------------------------------------------------------------------
    if dataset == 'papilledema':
        X, y = _call_first_available(
            ['load_papilledema_dataset',
             'load_papilledema',
             'load_papilledema_data'],
            img_size=(128, 128),
            max_samples_per_class=None)

        split = _first_present(['preprocess_medical_data',
                                'preprocess_data'])
        if split is None:
            raise AttributeError(
                "No preprocess_medical_data / preprocess_data in data_loader.")
        return split(X, y, test_size=0.15, val_size=0.15, random_state=42)

    # ------------------------------------------------------------------
    # DR: loader returns the 7-tuple author-provided split directly.
    # ------------------------------------------------------------------
    loader = _first_present(['load_diabetic_retinopathy_binary',
                             'load_dr_dataset',
                             'load_dr'])
    if loader is None:
        available = [x for x in dir(data_loader)
                     if not x.startswith('_') and
                     ('load' in x or 'dr' in x)]
        raise AttributeError(f"No DR loader found. Available: {available}")

    print(f"  using data_loader.{loader.__name__}")
    result = loader(img_size=(128, 128),
                    max_samples_per_class=None,
                    use_retino=False)

    # Case 1: already split by the loader (7-tuple) — use as-is.
    if isinstance(result, tuple) and len(result) == 7:
        return result

    # Case 2: generic (X, y) loader — split it here.
    if isinstance(result, tuple) and len(result) == 2:
        X, y = result
        split = _first_present(['preprocess_medical_data_binary',
                                'preprocess_medical_data'])
        if split is None:
            raise AttributeError(
                "No preprocess_medical_data_binary in data_loader.")
        return split(X, y, test_size=0.15, val_size=0.15, random_state=42)

    raise ValueError(
        f"Unexpected return shape from DR loader: {type(result)} "
        f"with length {len(result) if hasattr(result, '__len__') else '?'}.")


# ----------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------
def build_model(projection_dim, use_random_projection, regularization):
    return PRP_PIL(
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        projection_dim=projection_dim,
        n_hidden=2048,
        activation='tanh',
        use_advanced_features=True,
        use_attention=True,
        use_random_projection=use_random_projection,
        use_color_hist=False,
        use_gradients=True,
        regularization=regularization,
        random_state=42,
        batch_size=200)


def run_one(label, model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    t = time.time() - t0
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"  {label:40s}  acc = {acc * 100:.2f}%  ({t:.1f} s)")
    return dict(
        Configuration=label,
        Projection_dim=('none' if not model.use_random_projection
                        else model.projection_dim),
        Accuracy_pct=round(acc * 100, 2),
        Time_s=round(t, 1))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(dataset):
    print("=" * 74)
    print(f"Supplementary experiment — random projection information loss "
          f"— {dataset}")
    print(f"Patch size {PATCH_SIZE}×{PATCH_SIZE}  "
          f"(raw patch dim D = {D_RAW})")
    print("=" * 74)

    X_tr, X_va, X_te, y_tr, y_va, y_te, le = load_dataset(dataset)
    print(f"Train {X_tr.shape[0]}  Val {X_va.shape[0]}  "
          f"Test {X_te.shape[0]}\n")

    # Use the regularization λ matching each dataset's production config.
    lam = 1e-3 if dataset == 'papilledema' else 1e-5

    rows = []

    # --- (A) No-projection baseline ---
    print("(A) Baseline — raw patches fed directly to the hidden layer")
    rows.append(run_one(
        f"No projection (raw {D_RAW}-dim patches)",
        build_model(projection_dim=D_RAW,
                    use_random_projection=False,
                    regularization=lam),
        X_tr, y_tr, X_te, y_te))

    # --- (B) Projection-dimension sweep ---
    print("\n(B) Projection-dimension sweep")
    for L in L_VALUES:
        rows.append(run_one(
            f"Projection L = {L}",
            build_model(projection_dim=L,
                        use_random_projection=True,
                        regularization=lam),
            X_tr, y_tr, X_te, y_te))

    # --- Save CSV ---
    df = pd.DataFrame(rows)
    out_csv = f'Table_S1_L_sweep_{dataset}.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df.to_string(index=False))

    # --- Summary ---
    baseline = rows[0]['Accuracy_pct']
    l256 = next(r for r in rows if r['Projection_dim'] == 256)
    drop = baseline - l256['Accuracy_pct']

    print("\nSummary")
    print(f"  Raw {D_RAW}-dim baseline accuracy         : {baseline:.2f}%")
    print(f"  Projection L = 256 accuracy           : "
          f"{l256['Accuracy_pct']:.2f}%")
    print(f"  Information loss (768 → 256)          : {drop:.2f} pp")

    sweep = rows[1:]
    spread = (max(r['Accuracy_pct'] for r in sweep)
              - min(r['Accuracy_pct'] for r in sweep))
    print(f"  Spread across L ∈ {L_VALUES}: {spread:.2f} pp")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dr',
                    choices=['papilledema', 'dr'])
    args = ap.parse_args()
    np.random.seed(42)
    main(args.dataset)