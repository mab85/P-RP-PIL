# Binary differentiation task: true papilledema vs clinically relevant mimic
# Directly addresses Reviewer 4, Comment 1 ("differentiation of true 
# papilledema from clinically relevant mimics")

import numpy as np
import data_loader
from prp_elm import PRP_ELM
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_auc_score)
import time

# ---- Load the full 3-class Papilledema dataset ----
X, y = data_loader.load_papilledema_dataset(
    img_size=(128, 128), max_samples_per_class=None)
if X is None:
    raise SystemExit("Failed to load Papilledema dataset")

# ---- Keep only Papilledema and Pseudopapilledema ----
mask = np.isin(y, ['Papilledema', 'Pseudopapilledema'])
X = X[mask]; y = y[mask]

n_pap  = int((y == 'Papilledema').sum())
n_pseu = int((y == 'Pseudopapilledema').sum())
print(f"Binary task: Papilledema ({n_pap}) vs Pseudopapilledema ({n_pseu})")
print(f"Total: {len(X)}")

# ---- Split ----
X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
    data_loader.preprocess_medical_data(
        X, y, test_size=0.15, val_size=0.15, random_state=42)
print(f"Train {len(X_tr)}  Val {len(X_va)}  Test {len(X_te)}")
print(f"Class mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

# ---- Train the production P-RP-PIL model (same config as the paper) ----
model = PRP_ELM(
    patch_size=8, stride=2, projection_dim=256, n_hidden=2048,
    activation='tanh', use_advanced_features=True, use_attention=True,
    use_color_hist=False, use_gradients=True,
    regularization=1e-3, random_state=42, batch_size=200)

print("\nTraining ...")
t0 = time.time()
model.fit(X_tr, y_tr)
print(f"Trained in {time.time() - t0:.1f} s")

# ---- Evaluate ----
y_pred = model.predict(X_te)
y_proba = model.predict_proba(X_te)

acc = accuracy_score(y_te, y_pred)
cm  = confusion_matrix(y_te, y_pred, labels=[0, 1])

# Identify which column corresponds to 'Papilledema'
pap_idx = list(le.classes_).index('Papilledema')
y_true_bin = (y_te == pap_idx).astype(int)
y_prob_pap = y_proba[:, pap_idx]
auc = roc_auc_score(y_true_bin, y_prob_pap)

# Per-class metrics (Papilledema as the clinically positive class)
tp = int(((y_te == pap_idx) & (y_pred == pap_idx)).sum())
fp = int(((y_te != pap_idx) & (y_pred == pap_idx)).sum())
fn = int(((y_te == pap_idx) & (y_pred != pap_idx)).sum())
tn = int(((y_te != pap_idx) & (y_pred != pap_idx)).sum())

prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

print("\n" + "=" * 60)
print("Papilledema vs Pseudopapilledema — binary results")
print("=" * 60)
print(f"Accuracy        : {acc * 100:.2f}%")
print(f"AUC             : {auc:.4f}")
print(f"Precision (Pap.): {prec:.4f}")
print(f"Recall    (Pap.): {rec:.4f}")
print(f"Specificity(Pap.):{spec:.4f}")
print(f"F1        (Pap.): {f1:.4f}")
print(f"\nConfusion matrix (rows=true, cols=pred):")
print(f"  True Pap  -> Pred Pap: {tp}, Pred Pseu: {fn}")
print(f"  True Pseu -> Pred Pap: {fp}, Pred Pseu: {tn}")