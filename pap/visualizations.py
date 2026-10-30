import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, gc, warnings
warnings.filterwarnings('ignore')

import data_loader
from prp_pil import PRP_PIL


OUTPUT_DIR = 'visualization_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================================================================
def real_attention_map(model, image):
    
    ps = model.patch_sizes[0]
    st = model.strides[0]
    R  = model.R_list[0]

    H, W = image.shape[:2]
    weights = np.zeros((H, W), dtype=np.float32)
    counts  = np.zeros((H, W), dtype=np.float32)

    for i in range(0, H - ps + 1, st):
        for j in range(0, W - ps + 1, st):
            patch = image[i:i + ps, j:j + ps, :].flatten()
            z = np.tanh(patch @ R)
            v = float(z.var())
            weights[i:i + ps, j:j + ps] += v
            counts[i:i + ps, j:j + ps] += 1

    weights = weights / (counts + 1e-8)
    w = weights - weights.min()
    w = w / (w.max() + 1e-8)
    return w


# ======================================================================
def fig_attention(model, image, out_path):
    w = real_attention_map(model, image)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, alpha=0.55)
    im = ax.imshow(w, cmap='jet', alpha=0.45)
    ax.set_title(f'Attention Heatmap (Figure 10)\n'
                 f'patch {model.patch_sizes[0]}×{model.patch_sizes[0]}, '
                 f'stride {model.strides[0]}',
                 fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Attention weight')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f" Saved: {out_path}")


# ======================================================================
def fig_training_insights(csv_path, out_path):
    if not os.path.exists(csv_path):
        print(f" {csv_path} not found. Run perclass.py first."); return
    df = pd.read_csv(csv_path)
    df['Accuracy'] = df['Accuracy (%)'].astype(float) / 100.0
    df['Time'] = df['Training Time (s)'].astype(float)

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(df)))
    sizes = [400 if m == 'P-RP-PIL' else 250 for m in df['Model']]
    for (_, row), color, size in zip(df.iterrows(), palette, sizes):
        ax.scatter(row['Time'], row['Accuracy'], s=size,
                   color=color, edgecolors='black', alpha=0.85, zorder=3)
        ax.annotate(f"{row['Model']}\n({row['Accuracy']:.3f}, "
                    f"{row['Time']:.1f}s)",
                    (row['Time'], row['Accuracy']),
                    xytext=(8, 6), textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3',
                              fc='white', alpha=0.75))

    ax.set_xlabel('Training Time (seconds)', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Model Comparison: Accuracy vs Training Time',
                 fontweight='bold')
    ax.grid(alpha=0.3, zorder=0)
    ax.set_xlim(0, df['Time'].max() * 1.15)
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f" Saved: {out_path}")


# ======================================================================
def fig_architecture(out_path):
    steps = [
        ('Input Image\n(128×128×3)', '#cfe2f3'),
        ('Patch Extraction\n(8×8, stride=2)', '#b6d7f7'),
        ('Random Projection\n(dim=256)', '#9ec9e8'),
        ('Attention Pooling\n(Variance-based)', '#7fb4d8'),
        ('Hidden Layer\n(2048 units, tanh)', '#5c92bf'),
        ('Pseudo-Inverse Solve\n(λ=1e-3)', '#4a7cae'),
        ('Classification\n(3 classes)', '#3c679c'),
    ]
    fig, ax = plt.subplots(figsize=(8, 8))
    n = len(steps)
    for i, (label, color) in enumerate(steps):
        y = n - 1 - i
        ax.barh(y, 1.0, color=color, edgecolor='#333', linewidth=1.2,
                height=0.7)
        ax.text(0.5, y, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color='#111')
        ax.text(1.05, y, f'Step {i + 1}', ha='left', va='center',
                fontsize=9, color='#333')
        if i < n - 1:
            ax.annotate('', xy=(0.5, y - 0.4), xytext=(0.5, y - 0.2),
                        arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

    ax.set_xlim(0, 1.25)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('P-RP-PIL Architecture Overview', fontweight='bold',
                 fontsize=14, pad=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f" Saved: {out_path}")


# ======================================================================
def main():
    print("=" * 70)
    print("PAPILLEDEMA FIGURES 10, 11, 12 — production config")
    print("=" * 70)

    # Load and split
    X, y = data_loader.load_papilledema_dataset(
        img_size=(128, 128), max_samples_per_class=None)
    if X is None:
        print(" Could not load Papilledema dataset."); return
    X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
        data_loader.preprocess_medical_data(
            X, y, test_size=0.15, val_size=0.15, random_state=42)
    print(f"Train {X_tr.shape[0]} | Val {X_va.shape[0]} | "
          f"Test {X_te.shape[0]}")

    # Train production P-RP-PIL for the attention map
    print("\nTraining P-RP-PIL (production config) ...")
    prp = PRP_PIL(
        patch_size=8, stride=2, projection_dim=256,
        n_hidden=2048, activation='tanh',
        use_advanced_features=True, use_attention=True,
        use_color_hist=False, use_gradients=True,
        regularization=1e-3, random_state=42, batch_size=200)
    prp.fit(X_tr, y_tr)

    # Fig — real attention map on the first test image
    fig_attention(
        prp, X_te[0],
        os.path.join(OUTPUT_DIR, 'Figure_S5_Pap_Attn.png'))

    # Fig — training insights from cached comparison CSV
    fig_training_insights(
        'papilledema_model_comparison_results.csv',
        os.path.join(OUTPUT_DIR, 'Figure_S7_Pap_Train.png'))

    # Fig — architecture diagram with production config
    fig_architecture(os.path.join(OUTPUT_DIR, 'model_architecture.png'))

    print("\n Done. Figures written to visualization_output/")


if __name__ == '__main__':
    np.random.seed(42)
    main()