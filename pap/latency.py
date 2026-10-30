import time
import numpy as np
import data_loader
from prp_pil import PRP_PIL

# ---------------------------------------------------------------------
# CONFIG — set this per dataset
# ---------------------------------------------------------------------
DATASET = 'papilledema'   # change to 'dr' for the DR project
if DATASET == 'papilledema':
    MODEL_KWARGS = dict(
        patch_size=8, stride=2, projection_dim=256, n_hidden=2048,
        activation='tanh', use_advanced_features=True, use_attention=True,
        use_color_hist=False, use_gradients=True,
        regularization=1e-3, random_state=42, batch_size=200)
else:
    MODEL_KWARGS = dict(
        patch_size=6, stride=3, projection_dim=256, n_hidden=2048,
        activation='tanh', use_advanced_features=True, use_attention=True,
        use_color_hist=False, use_gradients=True,
        regularization=1e-5, random_state=42, batch_size=200)

# ---------------------------------------------------------------------
# Load and split
# ---------------------------------------------------------------------
if DATASET == 'papilledema':
    X, y = data_loader.load_papilledema_dataset(
        img_size=(128, 128), max_samples_per_class=None)
    X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
        data_loader.preprocess_medical_data(
            X, y, test_size=0.15, val_size=0.15, random_state=42)
else:
    X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
        data_loader.load_diabetic_retinopathy_binary(
            img_size=(128, 128), max_samples_per_class=None,
            use_retino=False)

print(f"Train {X_tr.shape[0]} | Val {X_va.shape[0]} | Test {X_te.shape[0]}")

# ---------------------------------------------------------------------
# Fit model
# ---------------------------------------------------------------------
print("Fitting P-RP-PIL ...")
model = PRP_PIL(**MODEL_KWARGS)
model.fit(X_tr, y_tr)

# ---------------------------------------------------------------------
# Warm-up (3 predictions), then time 100 single-image predictions
# ---------------------------------------------------------------------
for _ in range(3):
    _ = model.predict(X_te[:1])

N = 100
t0 = time.time()
for i in range(N):
    _ = model.predict(X_te[i:i + 1])
elapsed = time.time() - t0
print(f"\n{DATASET.upper()} — single-image inference latency: "
      f"{elapsed / N:.4f} s  (over {N} images)")