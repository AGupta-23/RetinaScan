# RetinaScan — Explainable Deep Learning for Diabetic Retinopathy Detection

**Progress Tracker**

**Currently On:** Phase 4, Task 4.1
**Last Updated:** 12th Aug
**Kaggle Notebooks Created So Far:** --> https://www.kaggle.com/code/abhidhagupta/retinascan-01-data/edit/run/341927887

---

## Project Summary

RetinaScan is an end-to-end deep learning pipeline that classifies retinal fundus images as **DR (Diabetic Retinopathy present)** or **No DR**, using transfer learning on a pretrained ResNet50. It covers the full pipeline — data collection, cleaning, preprocessing, EDA, training, evaluation, Grad-CAM explainability, and deployment as a live Streamlit demo. This is a simplified, binary-classification version (not the earlier 5-class RetinaVision AI approach) — built to be fully explainable and defensible in an interview or viva.

**Note:** This repo previously tracked a more complex 5-class severity-grading version (RetinaVision AI, with two-stage transfer learning and Quadratic Weighted Kappa). That approach was scoped down to keep the project simpler and easier to explain end-to-end — this tracker reflects the current, simplified plan.

---

## Phase 1: Project Setup

**What happens:** Create GitHub repo with a clean folder structure (`data/`, `notebooks/`, `src/`, `models/`, `app/`, `outputs/`). Set up a Kaggle account, verify GPU access. All datasets and heavy training stay on Kaggle — only code lives locally and gets pushed/pulled via `git clone`.

- [✅] 1.1 Create local project folder structure   --mkdir command 
- [✅] 1.2 Set up git + push skeleton to GitHub
- [✅] 1.3 Create Kaggle account + verify phone (for GPU access)
- [✅] 1.4 Locate/bookmark APTOS 2019 dataset on Kaggle
- [✅] 1.5 Set up local Python environment + requirements.txt  -- pip freeze
- [✅] 1.6 Write initial README.md

---

## Phase 2: Data Collection

**What happens:** Download the APTOS 2019 Blindness Detection dataset directly inside the Kaggle notebook — no local download needed. ~3,600 labeled retinal images across 5 severity classes (0–4).

- [✅] 2.1 Create first Kaggle Notebook (`retinascan-01-data`)
- [✅] 2.2 Enable GPU + internet, check quota
- [✅] 2.3 Load dataset inside notebook, verify file counts match `train.csv`
- [✅] 2.4 Save Kaggle notebook version

---

## Phase 3: Data Cleaning & Label Conversion

**What happens:** Check for corrupt/unreadable images and drop them. Convert 5-class severity labels into binary: class 0 → "No DR" (0), classes 1–4 → "DR" (1). Check class balance.

- [✅] 3.1 Scan for corrupt/unreadable images, drop them
- [✅] 3.2 Convert labels to binary (No DR / DR)
- [✅] 3.3 Check class balance (note % split here: No DR 49.29%, DR 50.71%)
- [✅] 3.4 Save clean CSV mapping filenames → binary labels

---

## Phase 4: Preprocessing

**What happens:** Resize all images to 224×224 (ResNet50 input size). Normalize using ImageNet mean/std. Stratified split into train (70%) / val (15%) / test (15%).

- [✅] 4.1 Write preprocessing script (resize + normalize)
- [✅] 4.2 Pull into Kaggle notebook (`retinascan-02-preprocessing`)
- [✅] 4.3 Visually validate preprocessing on samples
- [✅] 4.4 Create stratified train/val/test splits
- [✅] 4.5 Build and test Dataset/DataLoader class

---

## Phase 5: Exploratory Data Analysis (EDA)

**What happens:** Plot class distribution. Visualize sample images per class. Check image dimensions, color channels, and obvious quality issues.

- [ ] 5.1 Plot class distribution (DR vs No DR counts)
- [ ] 5.2 Visualize sample images from each class
- [ ] 5.3 Check image dimensions/quality issues
- [ ] 5.4 Save EDA figures locally + save Kaggle notebook version

---

## Phase 6: Model Building

**What happens:** Load ResNet50 pretrained on ImageNet. Replace final layer for binary output. Freeze early layers, unfreeze last few layers (single-stage fine-tuning).

- [ ] 6.1 Load pretrained ResNet50 (torchvision)
- [ ] 6.2 Replace final FC layer for binary output
- [ ] 6.3 Freeze early layers, unfreeze last few
- [ ] 6.4 Sanity-check forward pass on a batch

---

## Phase 7: Training (on Kaggle GPU)

**What happens:** Train using Binary Cross-Entropy loss + Adam optimizer over 10–20 epochs. Track train/val accuracy and loss per epoch. Save best checkpoint.

- [ ] 7.1 Create training Kaggle notebook (`retinascan-03-training`)
- [ ] 7.2 Set up loss (BCE) + optimizer (Adam)
- [ ] 7.3 Run training loop, log accuracy/loss per epoch
- [ ] 7.4 Watch for overfitting via val curves
- [ ] 7.5 Save best checkpoint (`best_model.pth`) — val result: _____
- [ ] 7.6 Download final model to local `models/` folder

---

## Phase 8: Evaluation

**What happens:** Run the trained model on the held-out test set. Compute accuracy, F1-score, precision, recall, confusion matrix. Analyze misclassifications.

- [ ] 8.1 Run inference on test set
- [ ] 8.2 Compute Accuracy/Precision/Recall/F1 — results: _____
- [ ] 8.3 Build + analyze confusion matrix
- [ ] 8.4 Review misclassified samples for patterns
- [ ] 8.5 Download metrics + confusion matrix locally

---

## Phase 9: Explainability with Grad-CAM

**What happens:** Apply Grad-CAM (`pytorch-grad-cam`) on sample test images to generate heatmaps showing which regions influenced each prediction. Overlay on original images, save side-by-side comparisons.

- [ ] 9.1 Write Grad-CAM script
- [ ] 9.2 Test on sample predictions in Kaggle notebook
- [ ] 9.3 Generate heatmap overlays (original vs. heatmap)
- [ ] 9.4 Save + download sample heatmap outputs

---

## Phase 10: Deployment & Documentation

**What happens:** Download trained model + sample images locally. Build Streamlit app: upload image → preprocess → predict → display result + confidence + Grad-CAM heatmap. Write final README with problem statement, dataset, pipeline, results, and how to run.

- [ ] 10.1 Choose interface: Streamlit
- [ ] 10.2 Build upload + preprocessing + inference flow
- [ ] 10.3 Display prediction + confidence score
- [ ] 10.4 Display Grad-CAM heatmap alongside original image
- [ ] 10.5 Test end-to-end locally
- [ ] 10.6 Finalize README.md (problem statement, dataset, pipeline, results, how to run, screenshots)
- [ ] 10.7 Write limitations section (regulatory, domain shift, no clinical validation)
- [ ] 10.8 Final GitHub cleanup

---

## Notes / Blockers Log

1. os library - file path issues
2. image file names --> never stored in .png form anywhere in csvs --> have to be converted thru os.path.join

-
