# Usage:
#   python roc_curve_generator.py --dataset papilledema
#   python roc_curve_generator.py --dataset dr
# ----------------------------------------------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import argparse
import data_loader
from prp_pil import PRP_PIL
from sklearn.metrics import (roc_curve, auc, roc_auc_score,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize


def build_model(dataset):
    if dataset == 'papilledema':
        return PRP_PIL(
            patch_size=8, stride=2, projection_dim=256, n_hidden=2048,
            activation='tanh', use_advanced_features=True, use_attention=True,
            use_color_hist=False, use_gradients=True, regularization=1e-3,
            random_state=42, batch_size=200)
    else:
        return PRP_PIL(
            patch_size=6, stride=1, projection_dim=512, n_hidden=2048,
            activation='tanh', use_advanced_features=True, use_attention=True,
            use_color_hist=False, use_gradients=True, regularization=1e-5,
            random_state=42, batch_size=200)


def main(dataset='papilledema'):
    print(f"\n=== Curve generation — {dataset} ===\n")

    # ---- Load and split ----
    if dataset == 'papilledema':
        X, y = data_loader.load_papilledema_dataset(
            img_size=(128,128), max_samples_per_class=800)
        X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
            data_loader.preprocess_medical_data(
                X, y, test_size=0.15, val_size=0.15, random_state=42)
    else:
        if hasattr(data_loader, 'combine_datasets_smart'):
            X, y = data_loader.combine_datasets_smart(
                img_size=(128,128), max_samples_per_class=800)
        else:
            X, y = data_loader.load_diabetic_retinopathy_binary(
                img_size=(128,128), max_samples_per_class=800,
                use_retino=False)
        X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
            data_loader.preprocess_medical_data_binary(
                X, y, test_size=0.15, val_size=0.15, random_state=42)

    n_classes = len(le.classes_)
    print(f"Classes: {list(le.classes_)}  |  Test set: {len(y_te)}")

    # ---- Train ----
    print("Training P-RP-PIL ...")
    model = build_model(dataset)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)

    # ==================================================================
    # Binary case (DR)
    # ==================================================================
    if n_classes == 2:
        # ---------- ROC ----------
        fpr, tpr, _ = roc_curve(y_te, proba[:, 1])
        auc_val = roc_auc_score(y_te, proba[:, 1])

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='#2E86AB', lw=2.5,
                 label=f'P-RP-PIL (AUC = {auc_val:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5,
                 linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve — DR Dataset', fontsize=13, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('Figure_S2_Pap_ROC.png', dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close()
        print(f" Saved: roc_curve.png      AUC = {auc_val:.4f}")

        # ---------- PR ----------
        precision, recall, _ = precision_recall_curve(y_te, proba[:, 1])
        ap = average_precision_score(y_te, proba[:, 1])

        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, color='#2E86AB', lw=2.5,
                 label=f'P-RP-PIL (AP = {ap:.4f})')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curve — DR Dataset',
                  fontsize=13, fontweight='bold')
        plt.legend(loc='lower left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('Figure_S1_Pap_PR.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f" Saved: precision_recall_curve.png  AP = {ap:.4f}")

    # ==================================================================
    # Multi-class case (Papilledema)
    # ==================================================================
    else:
        y_bin = label_binarize(y_te, classes=np.arange(n_classes))
        colors = ['#2E86AB', '#A23B72', '#F18F01']

        # ---------- ROC ----------
        plt.figure(figsize=(7, 6))
        aucs = []
        for i, cname in enumerate(le.classes_):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            a = auc(fpr, tpr)
            aucs.append(a)
            plt.plot(fpr, tpr, color=colors[i], lw=2.5,
                     label=f'{cname} (AUC = {a:.4f})')
        macro_auc = roc_auc_score(y_bin, proba,
                                  average='macro', multi_class='ovr')
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5,
                 linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(f'ROC Curve — Papilledema Dataset\n'
                  f'(Macro OvR AUC = {macro_auc:.4f})',
                  fontsize=13, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve1.png', dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close()
        print(f" Saved: roc_curve1.png")
        for cname, a in zip(le.classes_, aucs):
            print(f"   {cname:20s}: AUC = {a:.4f}")
        print(f"   Macro OvR AUC     : {macro_auc:.4f}")

        # ---------- PR ----------
        plt.figure(figsize=(7, 6))
        aps = []
        for i, cname in enumerate(le.classes_):
            precision, recall, _ = precision_recall_curve(
                y_bin[:, i], proba[:, i])
            ap = average_precision_score(y_bin[:, i], proba[:, i])
            aps.append(ap)
            plt.plot(recall, precision, color=colors[i], lw=2.5,
                     label=f'{cname} (AP = {ap:.4f})')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curve — Papilledema Dataset',
                  fontsize=13, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('precision_recall_curve1.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f" Saved: precision_recall_curve1.png")
        for cname, a in zip(le.classes_, aps):
            print(f"   {cname:20s}: AP = {a:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='papilledema',
                        choices=['papilledema', 'dr'])
    args = parser.parse_args()
    np.random.seed(42)
    main(args.dataset)