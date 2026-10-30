# dr/feature_importance.py — DR version
# ----------------------------------------------------------------------
# Generates Figure S10: empirical component importance for P-RP-PIL on DR.
#
# Reads:  Table_S6_DR_Ablation.csv  (produced by dr/ablation.py)
# Writes: Figure_S10_DR_Importance.png
# ----------------------------------------------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


CSV_PATH = 'Table_S6_DR_Ablation.csv'
OUT_PATH = 'Figure_S10_DR_Importance.png'
DATASET  = 'DR'
TAB_REF  = 'Table S6'


def build_figure(csv_path, out_path, dataset_label, table_ref):
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"{csv_path} not found. Run the ablation first: "
            f"python ablation.py --dataset dr")

    df = pd.read_csv(csv_path)
    print(f"Loaded: {csv_path}  ({len(df)} rows)")

    prod_mask = (df['Configuration'].str.startswith('w/o')
                 & ~df['Configuration'].str.contains('PIL', case=False))
    opt_mask  =  df['Configuration'].str.startswith('with')

    prod = df[prod_mask].copy()
    opt  = df[opt_mask].copy()

    if len(prod) + len(opt) == 0:
        raise SystemExit("No 'w/o X' or 'with X' rows in the CSV.")

    prod['Label'] = prod['Configuration'].str.replace('w/o ',  '', regex=False)
    opt['Label']  = opt['Configuration'].str.replace('with ', '', regex=False)

    prod['Effect'] = prod['Delta (pp)'].abs()
    prod['Kind']   = 'production'
    opt['Effect']  = opt['Delta (pp)']
    opt['Kind']    = 'optional'

    combined = pd.concat([
        prod[['Label', 'Effect', 'Kind']],
        opt[['Label', 'Effect', 'Kind']],
    ]).sort_values('Effect', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = ['#2E86AB' if k == 'production' else '#A23B72'
              for k in combined['Kind']]

    bars = ax.barh(combined['Label'], combined['Effect'],
                   color=colors, edgecolor='black',
                   linewidth=0.7, alpha=0.88)

    for bar, val in zip(bars, combined['Effect']):
        ha = 'left' if val >= 0 else 'right'
        offset = 0.35 if val >= 0 else -0.35
        ax.text(bar.get_width() + offset,
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.2f} pp', va='center', ha=ha,
                fontsize=9, fontweight='bold')

    ax.axvline(0, color='black', linewidth=1.3, zorder=0)
    ax.grid(axis='x', alpha=0.3, zorder=0)
    ax.set_xlabel(
        'Accuracy change (pp)\n'
        'Production: |Δ| when removed   ·   '
        'Optional: Δ when added (negative = hurts)',
        fontsize=10, fontweight='bold')
    ax.set_ylabel('Component', fontsize=10, fontweight='bold')
    ax.set_title(
        f'Empirical component importance — P-RP-PIL on {dataset_label}',
        fontsize=12, fontweight='bold', pad=12)

    legend_elements = [
        Patch(facecolor='#2E86AB', edgecolor='black',
              label='Used in production model (|Δ| when removed)'),
        Patch(facecolor='#A23B72', edgecolor='black',
              label='Not used (Δ when added; negative = hurts accuracy)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    effect_min = combined['Effect'].min()
    effect_max = combined['Effect'].max()
    pad = max(abs(effect_min), abs(effect_max)) * 0.20
    ax.set_xlim(effect_min - (pad if effect_min < 0 else 0),
                effect_max + (pad if effect_max > 0 else 0))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}\n")

    print(f"Computed component importance — {dataset_label}:")
    print(f"  {'Component':35s}  {'Effect':>10s}  Kind")
    print(f"  {'-'*35}  {'-'*10}  {'-'*15}")
    for _, row in combined.iterrows():
        kind_str = ('production' if row['Kind'] == 'production' else 'optional')
        print(f"  {row['Label']:35s}  {row['Effect']:+10.2f}  {kind_str}")
    print(f"\nReference for caption: {table_ref}")


if __name__ == '__main__':
    build_figure(CSV_PATH, OUT_PATH, DATASET, TAB_REF)