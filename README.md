# Paleographia Digitalis
## Automated Latin Manuscript Classification

Is an academic computer vision project, for the Machine Learning for the Humanities course of the DHDK master’s degree at University of Bologna, A.Y. 2025/2026, that bridges automated deep learning classification with traditional archival scholarship. It utilizes a high-performance weighted ensemble of modern neural network architectures to classify historical handwriting in medieval Latin manuscripts into 12 distinct paleographical traditions.

---

## Key Features
- **Advanced Ensemble Architecture:** Combines a tailored **ConvNeXt-Small** (capturing global semantic contexts and layouts) and **EfficientNet-V2-S** (specialized in fine-grained edge and stroke representations) via a 60/40 weighted probability soft-voting mechanism.
- **High-Resolution Processing:** Scales input resolution up to **$384 \times 384$ pixels** to successfully preserve high-density ink trajectories, ligatures, and *minim* structures.
- **Domain-Specific Regularization:** Incorporates class-weighted cross-entropy loss (mitigating imbalance across difficult categories like Types 6, 8, and 9) alongside manuscript-tailored augmentations (`RandomResizedCrop`, `ColorJitter`) while avoiding destructive geometric flips.
- **Paleographical Contextualization:** Classification errors and inconsistencies frequently arise due to the lack of standardization in the Middle Ages, during which several scripts intertwined with one another over time.
---

## Dataset (ICDAR 2017 CLaMM)
The model was trained on the ICDAR CLaMM (Classification of Latin Medieval Manuscripts) dataset,used in the ICFHR 2017 Competition on the Classification of Medieval Handwritings in Latin Script
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

## Setup for Reproducibility

Follow these steps to set up, run, and explore the project locally:

### 1. Download Repository and Dataset
* Clone or download this repository's `.zip` file and extract it.
* **Dataset:** Due to GitHub's storage limits, the high-resolution ICDAR 2017 CLaMM dataset is hosted externally on Kaggle. You can download it directly from the [Kaggle Dataset Link](https://www.kaggle.com/datasets/mariajgamboa/clammdst/).

### 2. Install Dependencies
Install all required Python libraries by running:
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebooks

- Navigate to the `notebooks/` directory.
- Run the notebooks sequentially, starting with the baseline.
- ⚠️ **Note:** Make sure to update the dataset root path variables in the notebook configuration cells to point to the local directory where you downloaded the dataset.

### 4. Start the Application
Once the model weights and environment are ready, launch the local server:
```bash
python app.py
```

Then, open **http://127.0.0.1:5001** in your browser to interact with the application.

### If not try the App online
Live Deployment: **https://sidereus.dev/paleographia-digitalis**
