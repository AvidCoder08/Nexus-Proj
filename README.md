# AI-Powered Fracture Detection in X-ray Images

Team: Shashank, Suprita, Srujan
Institution: PES University
Duration: 6 Weeks

## Executive Summary

This project builds a binary image classifier to flag musculoskeletal X-ray images as either
"Fractured" or "Normal". It uses transfer learning and common computer vision best practices
so radiologists can prioritize urgent cases and reduce diagnostic errors.

## Table of Contents

- Project Overview
- Setup & Requirements
- Datasets (recommended)
- Folder structure
- How to run (Colab / Local)
- Model development strategy
- Training & evaluation
- 6-week roadmap and week-by-week guide
- Resources & references

## Project Overview

Problem: Radiologists review hundreds of X-rays daily. An AI assistant can automatically detect
fractures, prioritize urgent cases, improve accuracy, and speed up workflows.

Goals:
- Binary classification: Fractured vs Normal
- Practical baseline: transfer learning (ResNet50 / DenseNet121)
- Target metrics: Accuracy >90%, Recall (sensitivity) prioritized

## Setup & Requirements

1. Clone this repository to your environment or open in Google Colab.
2. Create and activate a Python 3.8+ environment.
3. Install dependencies:

```
pip install -r requirements.txt
```

See `requirements.txt` for recommended packages.

## Recommended Datasets
- MURA (Stanford) — recommended for beginners: https://stanfordmlgroup.github.io/competitions/mura/
- FracAtlas — detailed annotations (localization/segmentation)
- Kaggle bone fracture datasets — quick prototyping

## Folder structure (starter)

```
/ (repo root)
├─ README.md
├─ requirements.txt
├─ week_by_week.md
├─ roadmap.md
├─ notebooks/Colab_Template.ipynb
├─ src/
│  ├─ data.py
│  ├─ model.py
│  ├─ train.py
│  ├─ evaluate.py (placeholder)
│  └─ infer.py (placeholder)
├─ app/
│  └─ streamlit_app.py
└─ presentation/
   └─ slides_notes.md
```

## How to run (quick start)

- Option A: Google Colab (recommended for free GPU)
  - Open `notebooks/Colab_Template.ipynb` in Colab.
  - Follow cells to mount Google Drive, download dataset, run preprocessing and training.

- Option B: Local
  - Create a virtualenv and install `requirements.txt`.
  - Prepare dataset folder in the expected layout (train/val/test with class subfolders).
  - Run training scaffold:

```
python src/train.py --data_dir /path/to/dataset --epochs 10
```

## Model & Training (high-level)

Use transfer learning with a pretrained backbone (ResNet50/DenseNet121). Freeze base layers first,
train the classification head, then selectively unfreeze and fine-tune.

Key callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.

## Evaluation & Interpretation

Compute accuracy, precision, recall, F1-score, AUC-ROC, and plot confusion matrix. Generate Grad-CAM
visualizations to interpret model attention.

## 6-Week Roadmap
See `week_by_week.md` and `roadmap.md` for detailed responsibilities and weekly deliverables.

## Resources
- TensorFlow/Keras docs: https://www.tensorflow.org
- MURA dataset paper: https://arxiv.org/abs/1712.06957
- Grad-CAM: Selvaraju et al.

## Next steps
1. Pick dataset and download (MURA recommended).
2. Run the Colab notebook or local scaffold.
3. Iterate on augmentation, model choice and hyperparameters.
4. Prepare demo and slides for submission.

---
Document Version: 1.0
Last Updated: November 2025
Prepared For: PES University B.Tech CSE Team
