
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import time
import data_loader
from prp_pil import PRP_PIL
from sklearn.metrics import accuracy_score


OUTPUT_CSV = 'Table_S1_L_sweep.csv'

# Fixed sweep design
PATCH_SIZE  = 16                            # raw patch dim D = 16*16*3 = 768
STRIDE      = 4
L_VALUES    = [128, 256, 384, 512, 768]
D_RAW       = PATCH_SIZE * PATCH_SIZE * 3   # = 768


# ----------------------------------------------------------------------
def load_papilledema():
    X, y = data_loader.load_papilledema_dataset(
        img_size=(128, 128), max_samples_per_class=None)
    X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
        data_loader.preprocess_medical_data(
            X, y, test_size=0.15, val_size=0.15, random_state=42)
    return X_tr, X_va, X_te, y_tr, y_va, y_te, le


def build_model(projection_dim, use_random_projection=True):
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
        regularization=1e-3,
        random_state=42,
        batch_size=200)


def run_one(label, model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    t = time.time() - t0
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"  {label:38s}  acc = {acc * 100:.2f}%  ({t:.1f} s)")
    return dict(
        Configuration=label,
        Projection_dim=('none' if not model.use_random_projection
                        else model.projection_dim),
        Accuracy_pct=round(acc * 100, 2),
        Time_s=round(t, 1))


# ----------------------------------------------------------------------
def main():
    print("=" * 74)
    print("Supplementary experiment — random projection information loss")
    print(f"Patch size {PATCH_SIZE}×{PATCH_SIZE}  (raw patch dim "
          f"D = {D_RAW})")
    print("=" * 74)

    X_tr, X_va, X_te, y_tr, y_va, y_te, le = load_papilledema()
    print(f"Train {X_tr.shape[0]}  Val {X_va.shape[0]}  "
          f"Test {X_te.shape[0]}\n")

    rows = []

    # --- (A) No-projection baseline: raw 768-dim patches ---
    print("(A) Baseline — raw patches fed directly to the hidden layer")
    rows.append(run_one(
        f"No projection (raw {D_RAW}-dim patches)",
        build_model(projection_dim=D_RAW, use_random_projection=False),
        X_tr, y_tr, X_te, y_te))

    # --- (B) Projection-dimension sweep ---
    print("\n(B) Projection-dimension sweep")
    for L in L_VALUES:
        rows.append(run_one(
            f"Projection L = {L}",
            build_model(projection_dim=L, use_random_projection=True),
            X_tr, y_tr, X_te, y_te))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(df.to_string(index=False))

    # --- Summary of information loss ---
    baseline = rows[0]['Accuracy_pct']
    l256 = next(r for r in rows if r['Projection_dim'] == 256)
    drop = baseline - l256['Accuracy_pct']
    print("\nSummary")
    print(f"  Raw {D_RAW}-dim baseline accuracy         : {baseline:.2f}%")
    print(f"  Projection L = 256 accuracy           : {l256['Accuracy_pct']:.2f}%")
    print(f"  Information loss (768 → 256)          : {drop:.2f} pp")

    print("\nFull spread across L ∈ {128, 256, 384, 512, 768}:")
    sweep = [r for r in rows[1:]]
    spread = max(r['Accuracy_pct'] for r in sweep) - \
             min(r['Accuracy_pct'] for r in sweep)
    print(f"  Range = {spread:.2f} pp  (max − min across the four L values)")


if __name__ == '__main__':
    np.random.seed(42)
    main()