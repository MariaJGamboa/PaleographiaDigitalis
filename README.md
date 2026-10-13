# Paleographia Digitalis: Automated Latin Manuscript Classification

Is an academic computer vision project, for the Machine Learning for the Humanities course of the DHDK master’s degree at University of Bologna, A.Y. 2025/2026, that bridges automated deep learning classification with traditional archival scholarship. It utilizes a high-performance weighted ensemble of modern neural network architectures to classify historical handwriting in medieval Latin manuscripts into 12 distinct paleographical traditions.

---

## Key Features
- **Advanced Ensemble Architecture:** Combines a tailored **ConvNeXt-Small** (capturing global semantic contexts and layouts) and **EfficientNet-V2-S** (specialized in fine-grained edge and stroke representations) via a 60/40 weighted probability soft-voting mechanism.
- **High-Resolution Processing:** Scales input resolution up to **$384 \times 384$ pixels** to successfully preserve high-density ink trajectories, ligatures, and *minim* structures.
- **Domain-Specific Regularization:** Incorporates class-weighted cross-entropy loss (mitigating imbalance across difficult categories like Types 6, 8, and 9) alongside manuscript-tailored augmentations (`RandomResizedCrop`, `ColorJitter`) while avoiding destructive geometric flips.
- **Paleographical Contextualization:** Accounts for historical manuscript variables such as morphological mixing (*mescidanza*) and the taxonomic 
---

## Performance Summary
Evaluated across a validation set of $N = 908$ medieval manuscript images, the final model achieved:
- **Overall Accuracy:** **80.40%**
- **Macro $F_1$-Score:** **0.81** (confirming balanced performance across minority and majority classes)


## Dataset (ICDAR 2017 CLaMM)
The dataset comprises 4,540 high-resolution manuscript scans categorized across 12 standard Latin script classes:
1. Caroline
2. Cursiva
3. Half-Uncial
4. Humanistic
5. Humanistic Cursive
6. Hybrida
7. Praegothica
8. Semihybrida
9. Semitextualis
10. Southern Textualis
11. Textualis
12. Uncial

---
