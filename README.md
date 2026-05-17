# HybridDenseViT — Liver Fibrosis Staging

Official implementation of:

> **"A Robust Hybrid CNN–Transformer Pipeline for Liver Fibrosis Staging:
> A Multi-Modal Validation on B-Mode Ultrasound and T2-Weighted MRI"**


---

## What this repository contains

| File | Description |
|---|---|
| `notebooks/HybridDenseViT_Ultrasound.ipynb` | Training pipeline for 5-class B-mode Ultrasound (METAVIR F0–F4) |
| `notebooks/HybridDenseViT_MRI.ipynb` | Training pipeline for 4-class T2-weighted MRI (METAVIR F0–F3) |
| `requirements.txt` | All Python dependencies |

---

## Results

| Dataset | Modality | Classes | QWK | Accuracy | Macro F1 |
|---|---|---|---|---|---|
| cirrMRI600+ | T2W-MRI | 4 (F0–F3) | **0.9801** | **97.02%** | **0.9720** |
| Liver Fibrosis Ultrasound Images | B-mode US | 5 (F0–F4) | **0.9922** | **97.94%** | **0.9707** |

- No clinically severe misclassifications in either modality
- All per-class AUC values above 0.997
- LayerCAM confirms complementary CNN and Transformer feature learning

---

## Model Architecture

HybridDenseViT combines two parallel branches:

- **DenseNet-121** — local texture feature extraction (F_CNN ∈ ℝ¹⁰²⁴)
- **ViT-B/16** — global structural modelling via self-attention (F_ViT ∈ ℝ⁷⁶⁸)
- **Concatenation** → single linear classification head (F_fused ∈ ℝ¹⁷⁹²)

The identical architecture is used for both modalities — only `num_classes` changes (5 for US, 4 for MRI).

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Weight decay | 1×10⁻⁴ |
| Learning rate | 1×10⁻⁴ |
| Batch size | 8 |
| Max epochs | 30 per fold |
| Early stopping | Patience = 5 (on validation QWK) |
| Cross-validation | 3-fold stratified |
| Image size | 224×224 |
| GPU | NVIDIA Tesla T4 (Kaggle) |

---

## Imbalance Handling

Dual-level strategy applied in both notebooks:
1. **WeightedRandomSampler** — inverse-frequency sample weights at the data level
2. **Focal Loss** — `alpha=1, gamma=2` at the loss level

---

## Preprocessing Pipeline

Same pipeline applied to both US and MRI:

```python
# Step 1 — 3×3 Median Filter (noise suppression)
img = cv2.medianBlur(img, 3)

# Step 2 — CLAHE on L-channel of LAB colourspace
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab[:, :, 0] = clahe.apply(lab[:, :, 0])
img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# Step 3 — Laplacian sharpening (centre coefficient = 9)
kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
img = cv2.filter2D(img, -1, kernel)
```

---

## Datasets

**Ultrasound:** Liver Fibrosis Ultrasound Images — Kaggle
https://www.kaggle.com/datasets/vibhingupta028/liver-histopathology-fibrosis-ultrasound-images
6,323 images | F0=2114, F1=861, F2=793, F3=857, F4=1698

**MRI:** cirrMRI600+ — Open Science Framework
https://osf.io/cuk24/
10,854 slices | F0=2185, F1=3280, F2=3483, F3=1906

---

## How to Run

### On Kaggle (recommended — free GPU)

1. Create a Kaggle account at kaggle.com
2. Click Create Notebook and upload the notebook file
3. Settings → Accelerator → GPU T4
4. Add the dataset and update the `data_dir` path in the notebook
5. Click Run All

### On Google Colab

1. Upload the notebook to colab.research.google.com
2. Runtime → Change runtime type → GPU
3. Run the installation cell at the top
4. Update `data_dir` to your dataset location
5. Run All

---

## Installation

```bash
pip install -r requirements.txt
```

---



---

## Acknowledgements

- [timm](https://github.com/rwightman/pytorch-image-models) — DenseNet-121 and ViT-B/16 pretrained weights
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — LayerCAM implementation
- [Albumentations](https://albumentations.ai/) — augmentation pipeline
- Kaggle — free GPU compute

---

