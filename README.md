[README.md](https://github.com/user-attachments/files/30386747/README.md)
# HybridDenseViT — Liver Fibrosis Staging

Official implementation of:

> **HybridDenseViT: A Unified Explainable Hybrid Deep Learning Framework for Multi-Stage Liver Fibrosis Staging Across T2-Weighted MRI and B-Mode Ultrasound**

---

## Overview

HybridDenseViT is a dual-branch hybrid CNN-Transformer architecture for automated, explainable, multi-stage liver fibrosis classification. The same feature extraction backbone — DenseNet-121 for local texture and ViT-B/16 for global structure — is applied to both T2-weighted MRI and B-mode ultrasound without modality-specific architectural modification.

Key properties:
- Patient-level cross-validation for MRI (zero inter-fold patient overlap verified)
- Ordinal-aware class-weighted loss for clinically appropriate imbalance correction
- LayerCAM (CNN branch) + Attention Rollout (ViT branch) for branch-specific explainability
- Ablation study with McNemar's significance testing confirming hybrid benefit on MRI

---

## Repository Contents

| File | Description |
|------|-------------|
| `notebooks/HybridDenseViT_MRI.ipynb` | Full training pipeline for 4-class T2W-MRI (METAVIR F0–F3), patient-level 5-fold CV |
| `notebooks/HybridDenseViT_Ultrasound.ipynb` | Full training pipeline for 5-class B-mode Ultrasound (METAVIR F0–F4), image-level 5-fold CV |
| `requirements.txt` | Python dependencies |

---

## Results

| Dataset | Modality | Classes | OOF QWK | Accuracy | Macro F1 | Patient QWK |
|---------|----------|---------|---------|----------|----------|-------------|
| cirrMRI600+ | T2W-MRI | 4 (F0–F3) | 0.9154 [0.8808–0.9473] | 85.99% | 0.8824 | **0.9406** |
| Liver Fibrosis Ultrasound | B-mode US | 5 (F0–F4) | 0.9958 [0.9930–0.9986] | 98.83% | 0.9830 | N/A |

Values in brackets are 95% confidence intervals (Student's t-distribution, 4 d.f.).

**MRI evaluated under patient-level StratifiedGroupKFold (5-fold) — zero patient overlap across folds.**

No non-adjacent misclassifications in either modality.

### Ablation Study (MRI — Patient-Level 5-Fold)

| Model | QWK mean ± std | QWK 95% CI | Acc mean ± std |
|-------|---------------|-----------|---------------|
| DenseNet-121 Only | 0.8695 ± 0.0296 | [0.8284, 0.9106] | 0.7941 ± 0.0338 |
| ViT-B/16 Only | 0.9056 ± 0.0242 | [0.8721, 0.9391] | 0.8447 ± 0.0295 |
| **HybridDenseViT (Proposed)** | **0.9141 ± 0.0239** | **[0.8808, 0.9473]** | **0.8596 ± 0.0285** |

McNemar's test: Hybrid vs DenseNet-121 p = 5.55×10⁻⁴³ | Hybrid vs ViT-B/16 p = 1.97×10⁻⁷

---

## Model Architecture

HybridDenseViT processes each input through two parallel branches:

- **DenseNet-121 branch** — local parenchymal texture extraction via dense skip connections → Global Average Pooling → $F_{\text{CNN}} \in \mathbb{R}^{1024}$
- **ViT-B/16 branch** — global morphological structure modelling via multi-head self-attention over 196 patch tokens → [CLS] token → $F_{\text{ViT}} \in \mathbb{R}^{768}$
- **Fusion head** — Concat($F_{\text{CNN}}$, $F_{\text{ViT}}$) → BN → Dropout(0.5) → Linear(1792→512) → GELU → Dropout(0.4) → Linear(512→128) → GELU → Dropout(0.4) → Linear(128→C)

C = 4 for MRI | C = 5 for Ultrasound

---

## Training Configuration

| Hyperparameter | T2W-MRI | B-Mode US |
|----------------|---------|-----------|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) | AdamW (β₁=0.9, β₂=0.999) |
| Weight Decay | 2×10⁻² | 1×10⁻⁴ |
| LR (fusion head) | 5×10⁻⁵ | 1×10⁻⁴ |
| LR (DenseNet last block) | 8×10⁻⁶ | 1×10⁻⁴ |
| LR (ViT last 3 blocks) | 5×10⁻⁶ | 1×10⁻⁴ |
| LR Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Physical Batch Size | 16 | 8 |
| Effective Batch Size | 32 (grad. accum. ×2) | 8 |
| Max Epochs | 50 per fold | 40 per fold |
| Early Stopping | Patience = 15 (val. QWK) | Patience = 8 (val. QWK) |
| Gradient Clipping | Max norm = 0.3 | Not applied |
| Fine-tuning Protocol | 3-phase curriculum freeze | Full fine-tuning |
| Label Smoothing | ε = 0.10 | ε = 0.05 |
| Dropout (fusion/FC) | 0.5 / 0.4 | 0.3 / 0.2 |
| Cross-Validation | Patient-level StratifiedGroupKFold (5-fold) | Image-level StratifiedKFold (5-fold) |
| Random Seed | 42 | 42 |
| GPU | NVIDIA Tesla T4 (Kaggle) | NVIDIA Tesla T4 (Kaggle) |

---

## Class Imbalance Correction

An Ordinal Cross-Entropy Loss combining class-frequency weighting with an ordinal penalty is applied:

$$\mathcal{L} = \mathcal{L}_{\text{CE}}(\hat{y}, y) + \lambda \cdot \mathcal{L}_{\text{ord}}(\hat{y}, y)$$

- **Class-weighted CE:** $\omega_c = N / (C \cdot n_c)$ with additional ×2 boost on F2 class (MRI only)
- **Ordinal penalty:** MSE between soft predicted ordinal index and true label (λ = 0.3)
- **Label smoothing:** ε = 0.10 (MRI), ε = 0.05 (US)

No Weighted Random Sampler is used. Imbalance correction is achieved entirely within the loss function.

---

## Preprocessing Pipeline

The same three-step pipeline is applied identically to both modalities:

```python
import cv2
import numpy as np

def preprocess(img):
    # Step 1 — 3×3 Median Filter (noise suppression)
    img = cv2.medianBlur(img, 3)

    # Step 2 — CLAHE on L-channel of LAB colour space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Step 3 — Resize and normalise
    img = cv2.resize(img, (224, 224))
    return img

# Normalisation
# T2W-MRI:  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  (ImageNet)
# B-mode US: mean=[0.5, 0.5, 0.5],       std=[0.5, 0.5, 0.5]       (symmetric)
```

> **Note:** Unsharp masking (sharpening) used in an earlier version of this pipeline has been removed. High-pass sharpening re-amplifies high-frequency noise in the same spectral band suppressed by the median filter, which degrades parenchymal texture signal critical for distinguishing adjacent fibrosis stages.

---

## Explainability

| Branch | Method | Target |
|--------|--------|--------|
| DenseNet-121 | LayerCAM | `features.norm5` (final normalisation layer) |
| ViT-B/16 | Attention Rollout | All 12 encoder blocks (`blk.attn`) |

Attention Rollout is used for the ViT branch because gradient-based CAM methods produce spatially unreliable attributions on transformer patch tokens.

---

## Datasets

**B-mode Ultrasound:**
Liver Histopathology Fibrosis Ultrasound Images (Kaggle)
https://www.kaggle.com/datasets/vibhingupta028/liver-histopathology-fibrosis-ultrasound-images
6,323 images | F0=2,114 | F1=861 | F2=793 | F3=857 | F4=1,698

**T2-Weighted MRI:**
cirrMRI600+ (Open Science Framework)
https://osf.io/cuk24/
6,964 axial slices | 305 patients | F0=859 | F1=2,614 | F2=2,239 | F3=1,252

---

## How to Run

### On Kaggle (recommended — free GPU)

1. Create a Kaggle account at kaggle.com
2. Click **Create Notebook** and upload the `.ipynb` file
3. Settings → Accelerator → **GPU T4 x2**
4. Add the dataset and update the `data_dir` path in the notebook
5. Click **Run All**

### On Google Colab

1. Upload the notebook to colab.research.google.com
2. Runtime → Change runtime type → **GPU**
3. Run the installation cell at the top
4. Update `data_dir` to your dataset location
5. **Run All**

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `timm`, `albumentations`, `nibabel`, `scikit-learn`, `opencv-python`, `matplotlib`, `numpy`

---

## Citation

If you use this code or results in your work, please cite:

```
@article{hybridensevit2025,
  title   = {HybridDenseViT: A Unified Explainable Hybrid Deep Learning Framework
             for Multi-Stage Liver Fibrosis Staging Across T2-Weighted MRI
             and B-Mode Ultrasound},
  author  = {Rhidi, Raufu Umme and Jibon, Ferdaus Anam and Shahriar, Farhan
             and Islam, Muradul and Kamal, A. H. M. and Hossain, Gahangir},
  journal = {[Journal Name]},
  year    = {2025}
}
```

---

## Acknowledgements

- [timm](https://github.com/huggingface/pytorch-image-models) — DenseNet-121 and ViT-B/16 pretrained weights
- [Albumentations](https://albumentations.ai/) — augmentation pipeline
- [NiBabel](https://nipy.org/nibabel/) — NIfTI file handling for MRI
- [Kaggle](https://www.kaggle.com/) — free GPU compute
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — LayerCAM implementation

---

## License

This project is licensed under the MIT License.
