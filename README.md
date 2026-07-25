# Brain Tumor MRI Classification with Cross-Dataset Generalization & Grad-CAM

> **Undergraduate Research Paper | COMSATS University Islamabad, Sahiwal Campus | 2026**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper: In Progress](https://img.shields.io/badge/Paper-IEEE%20Format-red)]()

---

## 🧠 Overview

This repository contains the complete implementation of an explainable AI framework for brain tumor classification from MRI images using transfer learning, with a **novel cross-dataset generalization evaluation** not previously reported in existing literature.

**Key Question:** Can a model trained on one brain tumor MRI dataset generalize to a completely different, unseen dataset — without any retraining?

**Answer:** Yes — with only 8.31% accuracy drop.

---

## 🏆 Results

### Primary Dataset (Kaggle Brain Tumor MRI, 7,023 images)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| ResNet50 | 71.06% | 73.87% | 71.06% | 65.63% |
| EfficientNetB0 | 25.00% | 6.25% | 25.00% | 10.00% |
| **MobileNetV2** | **94.44%** | **94.70%** | **94.44%** | **94.32%** |

### Cross-Dataset Generalization (Figshare, 3,064 unseen images)

| Dataset | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| Kaggle (Primary) | 94.44% | 94.70% | 94.44% | 94.32% |
| Figshare (External) | 86.13% | 86.75% | 86.13% | 85.82% |
| **Generalization Gap** | **8.31%** | — | — | — |

> 📌 **Novel Contribution:** No existing paper has reported cross-dataset generalization between these two specific datasets. Most papers only evaluate on a single train-test split.

---

## 🔥 Novel Contributions

1. **Cross-dataset generalization testing** — trained on Kaggle, tested on Figshare (3,064 unseen MRI images) without retraining
2. **Grad-CAM explainability** — heatmaps showing WHERE the model looks inside MRI scans for each tumor class
3. **Architectural failure analysis** — systematic investigation of EfficientNetB0 collapse under partial fine-tuning
4. **Lightweight model superiority** — MobileNetV2 (3.4M params) outperforms ResNet50 (25.6M params) by 23.38%

---

## 📊 Grad-CAM Visualizations

The model correctly localizes tumor regions across all 4 classes:
- **Glioma** → peripheral cerebral regions
- **Meningioma** → tumor mass along brain surface  
- **Pituitary** → central sellar region (99.7% confidence)
- **No Tumor** → diffuse, distributed attention

---

## 🗂️ Dataset

| Dataset | Images | Classes | Role |
|---------|--------|---------|------|
| [Kaggle Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | 7,023 | 4 (Glioma, Meningioma, No Tumor, Pituitary) | Training + Testing |
| [Figshare Brain Tumor](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) | 3,064 | 3 (Glioma, Meningioma, Pituitary) | External Validation Only |

---

## 🛠️ Tech Stack

- **Python** 3.10
- **TensorFlow / Keras** 2.x
- **scikit-learn** — metrics evaluation
- **NumPy / Pandas** — data handling
- **Matplotlib** — visualization
- **Grad-CAM** — explainability
- **Google Colab** — T4 GPU training
- **Overleaf** — IEEE paper formatting

---

## 📁 Repository Structure
brain-tumor-cross-dataset-classification/
│
├── brain_tumor_classification.ipynb # Main training pipeline (all 3 models)
├── gradcam_crossdataset_eval.ipynb # Grad-CAM + cross-dataset evaluation
├── exploratory_analysis.ipynb # Dataset exploration and visualization
├── sample_images.png # Figure 1 — MRI sample grid
├── gradcam_results.png # Figure 2 — Grad-CAM heatmaps
├── requirements.txt # Python dependencies
└── README.md

---

## 🚀 How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/ekrasafdar/brain-tumor-cross-dataset-classification.git
cd brain-tumor-cross-dataset-classification
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Download datasets:**
- [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)

**4. Open in Google Colab:**
- Upload `brain_tumor_classification.ipynb` to Google Colab
- Enable GPU: Runtime → Change runtime type → T4 GPU
- Mount Google Drive and update the dataset path
- Run all cells

**5. For Grad-CAM + Cross-Dataset Evaluation:**
- Open `gradcam_crossdataset_eval.ipynb`
- Load your trained MobileNetV2 model from Drive
- Run all cells

---

## 📄 Paper

**Title:** Cross-Dataset Generalization and Explainability Analysis of Transfer Learning Models for Brain Tumor MRI Classification

**Author:** Iqra Safdar — COMSATS University Islamabad, Sahiwal Campus

**Format:** IEEE Conference Paper

**Status:** 🔄 Under supervisor review — arXiv submission pending

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| TensorFlow / Keras | Model training |
| scikit-learn | Evaluation metrics |
| NumPy / Pandas | Data handling |
| Matplotlib / Pillow | Visualization |
| Grad-CAM | Explainability |
| Google Colab (T4 GPU) | Training environment |
| Overleaf | IEEE paper formatting |
| GitHub | Version control |

---

## 👩‍💻 Author

**Iqra Safdar**
5th Semester, BS Computer Science
COMSATS University Islamabad (CUI), Sahiwal Campus
📧 sp24-bcs-205@student.cuisahiwal.edu.pk
🔗 [GitHub](https://github.com/ekrasafdar)

---

## 📜 License

MIT License — free to use with attribution.

---

## ⭐ If this research helped you, consider starring the repo!
