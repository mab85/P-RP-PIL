# P-RP-PIL

Code for the paper:

> **P-RP-PIL: A Novel Patch-Based Random Projection Pseudo-Inverse
> Learning Framework for Automated Retinal Disease Detection in
> Fundus Images**  
> Mohammed A. B. Mahmoud, Arab Open University, Riyadh, KSA.

This repository reproduces every table (I–VIII) and figure (1–6) in the main text, and provides full reproduction scripts for all supplementary tables (S1–S7) and figures (S1–S10). 

---

## 1. Install

```bash
pip install -r requirements.txt
```


### Step 2
```markdown
## 2. Download the datasets

Place both datasets under `data/`:

```text
data/
|-- Normal/
|-- Papilledema/
|-- Pseudopapapilledema/
`-- Diagnosis of Diabetic Retinopathy/
    |-- train/{DR,No_DR}/
    |-- valid/{DR,No_DR}/
    `-- test/{DR,No_DR}/
```

- Papilledema: https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/Papilledema.md
- DR: https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy

---


### Step 3
```markdown
## 3. Reproduce all Papilledema results

```bash
cd pap
python per_class.py              # Tables I, II + Figures 2, 3
python mimic_differentiation.py  # Section IV.A binary mimic analysis
python ablation.py               # Table V
python multi_seed.py             # Table VI (Pap) + Supplementary Table S2
python complexity.py             # Table VII
python L_sweep.py                # Supplementary Table S1
python robustness.py             # Supplementary Table S3
python patch_sweep.py            # Supplementary Table S4
python roc_pr.py                 # Supplementary Figures S1, S2
python visualizations.py         # Figure 6 + Supplementary Figures S5, S7, S9
python latency.py                # Inference latency (Papilledema)
```

### Step 4
```markdown
## 4. Reproduce all DR results

```bash
cd dr
python per_class.py              # Tables III, IV + Figures 4, 5
python ablation.py               # Supplementary Table S6
python multi_seed.py             # Table VIII + Supplementary Table S7
python complexity.py             # Table VIII
python L_sweep.py                # Supplementary Table S1
python robustness.py             # Supplementary Table S3
python patch_sweep.py            # Supplementary Table S5
python roc_pr.py                 # Supplementary Figures S3, S4
python visualizations.py         # Supplementary Figures S6, S8, S10
python latency.py                # Inference latency (DR)           
```

---


## 5. Complete artifact map

### Main Text Tables & Figures (inside the 14-page PDF)

| Paper artifact | Script | Command |
|---|---|---|
| Table I - Pap main comparison | `pap/main_comparison.py` | `cd pap && python main_comparison.py` |
| Table II - Pap per-class | `pap/per_class.py` | `cd pap && python per_class.py` |
| Table III - DR main comparison | `dr/main_comparison.py` | `cd dr && python main_comparison.py` |
| Table IV - DR per-class | `dr/per_class.py` | `cd dr && python per_class.py` |
| Table V - Pap ablation | `pap/ablation.py` | `cd pap && python ablation.py --dataset papilledema` |
| Table VI - Pap extended metrics | `pap/multi_seed.py` | `cd pap && python multi_seed.py --dataset papilledema` |
| Table VII - Pap complexity | `pap/complexity.py` | `cd pap && python complexity.py --dataset papilledema` |
| Table VIII - DR complexity | `dr/complexity.py` | `cd dr && python complexity.py --dataset dr` |
| Figure 1 - Architecture schematic | Provided in repo | `document(1).png` |
| Figure 2 - Pap per-class comparison | `pap/per_class.py` | `papilledema_per_class_performance.png` |
| Figure 3 - Pap confusion matrices | `pap/per_class.py` | `visualization_output/comparison_confusion_matrices.png` |
| Figure 4 - DR per-class comparison | `dr/visualizations.py` | `visualization_output_dr/per_class_performance_visualization.png` |
| Figure 5 - DR confusion matrices | `dr/visualizations.py` | `visualization_output_dr/comparison_confusion_matrices_dr.png` |
| Figure 6 - Pipeline schematic | `pap/visualizations.py` | `visualization_output/model_architecture.png` |

### Supplementary Tables & Figures (compiled in `supplementary/supplementary_material.pdf`)

| Paper artifact | Script | Output file |
|---|---|---|
| Supp. Table S1 - L sweep | `pap/L_sweep.py` + `dr/L_sweep.py` | `Table_S1_L_sweep_*.csv` |
| Supp. Table S2 - Multi-seed | `pap/multi_seed.py` + `dr/multi_seed.py` | `Table_S2_Multi_Seed_*.csv` |
| Supp. Table S3 - Robustness | `pap/robustness.py` + `dr/robustness.py` | `Table_S3_Robustness_*.csv` |
| Supp. Table S4 - Pap patch sweep | `pap/patch_sweep.py` | `Table_S4_Pap_Sweep.csv` |
| Supp. Table S5 - DR patch sweep | `dr/patch_sweep.py` | `Table_S5_DR_Sweep.csv` |
| Supp. Table S6 - DR ablation | `dr/ablation.py` | `Table_S6_DR_Ablation.csv` |
| Supp. Table S7 - DR extended metrics | `dr/multi_seed.py` | `Table_S2_Multi_Seed_dr.csv` |
| Supp. Fig S1 - Pap precision-recall | `pap/roc_pr.py` | `Figure_S1_Pap_PR.png` |
| Supp. Fig S2 - Pap ROC | `pap/roc_pr.py` | `Figure_S2_Pap_ROC.png` |
| Supp. Fig S3 - DR precision-recall | `dr/roc_pr.py` | `Figure_S3_DR_PR.png` |
| Supp. Fig S4 - DR ROC | `dr/roc_pr.py` | `Figure_S4_DR_ROC.png` |
| Supp. Fig S5 - Pap attention heatmap | `pap/visualizations.py` | `visualization_output/attention_heatmap_figure10.png` |
| Supp. Fig S6 - DR attention heatmap | `dr/visualizations.py` | `visualization_output_dr/attention_heatmap_dr.png` |
| Supp. Fig S7 - Pap training insights | `pap/visualizations.py` | `visualization_output/training_insights.png` |
| Supp. Fig S8 - DR training insights | `dr/visualizations.py` | `visualization_output_dr/training_insights_dr.png` |
| Supp. Fig S9 - Pap component importance | `pap/feature_importance.py` | `Figure_S9_Pap_Importance.png` |
| Supp. Fig S10 - DR component importance | `dr/feature_importance.py` | `Figure_S10_DR_Importance.png` |

### Auxiliary outputs (Section IV.A and Section IV.C)

| Paper text | Script | Output |
|---|---|---|
| Section IV.A binary mimic analysis | `pap/mimic_differentiation.py` | printed to console |
| Inference latency paragraph | `pap/latency.py`, `dr/latency.py` | printed to console |
| DenseNet-121 latency comparison | `pap/latency.py` | printed to console |

## 6. Folder structure

```text
p-rp-pil/
|-- README.md
|-- requirements.txt
|-- data/                     <- download datasets here (gitignored)
|   |-- Normal/
|   |-- Papilledema/
|   |-- Pseudopapapilledema/
|   `-- Diagnosis of Diabetic Retinopathy/
|       |-- train/{DR,No_DR}/
|       |-- valid/{DR,No_DR}/
|       `-- test/{DR,No_DR}/
|-- pap/                      <- Papilledema pipeline
|   |-- data_loader.py        <- Papilledema loader
|   |-- prp_pil.py            <- P-RP-PIL model
|   |-- main_comparison.py
|   |-- per_class.py
|   |-- mimic_differentiation.py
|   |-- ablation.py
|   |-- L_sweep.py
|   |-- multi_seed.py
|   |-- complexity.py
|   |-- robustness.py
|   |-- patch_sweep.py
|   |-- roc_pr.py
|   |-- visualizations.py
|   |-- feature_importance.py
|   |-- latency.py
|   |-- densenet_latency.py
|   `-- baseline.py           <- reference only; per_class.py is preferred
|-- dr/                       <- DR pipeline
|   |-- data_loader.py        <- DR loader
|   |-- prp_pil.py            <- P-RP-PIL model
|   |-- main_comparison.py
|   |-- per_class.py
|   |-- ablation.py
|   |-- L_sweep.py
|   |-- multi_seed.py
|   |-- complexity.py
|   |-- robustness.py
|   |-- patch_sweep.py
|   |-- roc_pr.py
|   |-- visualizations.py
|   |-- feature_importance.py
|   `-- latency.py
`-- supplementary/            <- Supplementary Material
    |-- supplementary_material.pdf
    `-- (compiled figures and CSVs for S1-S10)

