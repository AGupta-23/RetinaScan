# RetinaScan — Explainable Deep Learning for Diabetic Retinopathy Detection

## Project Overview

RetinaScan is an end-to-end deep learning pipeline that classifies retinal fundus images as either **DR (Diabetic Retinopathy present)** or **No DR**, using transfer learning on a pretrained CNN. The project goes beyond a standalone model — it covers the complete data science lifecycle: data collection, cleaning, preprocessing, exploratory analysis, model training, evaluation, explainability, and deployment as a live interactive demo.

The core idea: a user uploads a retinal (fundus) eye image, and the system returns a prediction ("DR" or "No DR") with a confidence score, along with a Grad-CAM heatmap showing which regions of the eye image the model focused on to make that decision.

This is a **student/portfolio-level proof-of-concept**, not a certified medical device. It demonstrates the feasibility and workflow of AI-assisted DR screening, not a deployment-ready clinical tool.

---

## Problem Statement

Diabetic Retinopathy is a diabetes complication that damages blood vessels in the retina and is a leading cause of preventable blindness. Early detection through retinal screening is effective, but manual screening requires trained ophthalmologists, who are in short supply relative to the diabetic population — particularly in countries like India. This project explores whether a deep learning model can automatically flag DR presence from a retinal image, with visual explanations to support (not replace) clinical judgment.

---

## Dataset

- **Source:** APTOS 2019 Blindness Detection dataset (public, Kaggle)
- **Content:** ~3,600 labeled retinal fundus images
- **Original labels:** 5-class severity scale (0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR)
- **Simplification:** Labels collapsed into binary classification — class 0 → "No DR", classes 1–4 → "DR" — to keep the problem well-scoped, explainable, and free of the complications of ordinal, imbalanced multi-class classification.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Data handling | pandas, numpy |
| Image processing | OpenCV, Pillow |
| Deep learning framework | PyTorch + torchvision |
| Model architecture | ResNet50 (ImageNet-pretrained, fine-tuned) |
| Evaluation | scikit-learn (accuracy, F1, precision, recall, confusion matrix) |
| Explainability | pytorch-grad-cam (Grad-CAM heatmaps) |
| Compute | Kaggle Notebooks (free GPU) |
| Deployment | Streamlit (local interactive web app) |
| Version control | GitHub (also used as the code bridge into Kaggle via `git clone`) |

**Workflow constraint driving the architecture:** Development happens on a storage-limited laptop, so all heavy compute (training, GPU work) runs on Kaggle Notebooks. The laptop only holds code, which is pushed to GitHub and pulled into Kaggle at runtime. Datasets and large model weights never touch the laptop — only the final trained model file and small output artifacts (metrics, plots, sample heatmaps) are downloaded locally to power the demo app.

---

## Machine Learning Approach

**Why deep learning:** Unlike classical ML, where features must be hand-engineered, a Convolutional Neural Network (CNN) learns hierarchical visual features directly from raw pixels. Early layers learn simple patterns (edges, color gradients), middle layers combine these into shapes (vessel structures, spots), and deep layers learn complex, task-specific patterns associated with DR (hemorrhages, exudates, vessel damage) — all learned automatically from labeled data via backpropagation.

**Why transfer learning:** Training a CNN from scratch requires far more data and compute than is available here. Instead, ResNet50 — pretrained on ImageNet (millions of general images) — is used as a starting point. Its early layers (general visual features like edges and textures) are frozen, while the last few layers are unfrozen and fine-tuned specifically on retinal images. This is the standard, defensible approach for applying deep learning with limited data and resources.

**Training details:**
- Loss function: Binary Cross-Entropy
- Optimizer: Adam
- Epochs: modest range (10–20), monitored for overfitting via train/validation loss and accuracy curves
- Best checkpoint saved based on validation performance (`best_model.pth`)

---

## Evaluation

The trained model is evaluated on a held-out test set using:
- Accuracy
- F1-score
- Precision and Recall
- Confusion matrix (to inspect false positives/negatives)

Misclassifications are analyzed for patterns (e.g., borderline mild DR cases being harder to classify), which feeds into an honest discussion of the model's limitations.

---

## Explainability — Grad-CAM

A trained deep learning model is normally a "black box" — it outputs a prediction with no visible reasoning, which is a serious trust problem in any medical-adjacent context. **Grad-CAM (Gradient-weighted Class Activation Mapping)** addresses this by generating a heatmap over the input image, highlighting which pixel regions most influenced the model's prediction.

In this project, Grad-CAM heatmaps are generated for sample test images and overlaid on the original retinal images, so that predictions are accompanied by a visual explanation (e.g., "the model flagged DR because it focused on this hemorrhage-like region"), turning the system from a black-box classifier into an interpretable one.

---

## Deployment

The final trained model (`.pth` file, small enough for a local machine) is downloaded from Kaggle and integrated into a **Streamlit web app**:

1. User uploads a retinal fundus image
2. App preprocesses the image (resize, normalize) to match training conditions
3. Model returns a prediction (DR / No DR) with a confidence score
4. Grad-CAM heatmap is displayed alongside the original image

This provides a live, interactive demo — a working artifact that is far more compelling than a notebook of code alone.

---

## Documentation

A GitHub repository hosts the full project with a structured README covering: problem statement, dataset description, pipeline overview, results/metrics, instructions to run the Streamlit app locally, and screenshots/GIFs of the demo in action.

---

## Real-World Context & Limitations (honest framing)

This project is a **proof-of-concept**, not a clinically deployable tool. Important limitations to acknowledge:

- **No regulatory approval:** Deployment in real hospitals would require CDSCO approval (India's medical device regulator) as the tool falls under Software as a Medical Device (SaMD) classification.
- **Domain shift risk:** The model is trained on APTOS 2019 images collected under specific conditions; real hospital images (different cameras, lighting, patient populations) may differ enough to hurt performance — a well-known generalization challenge in medical imaging AI.
- **No clinical validation:** Metrics are computed on a held-out split of the same dataset, not validated against real patient outcomes or reviewed by ophthalmologists.
- **Not a diagnostic replacement:** The system is designed as a screening-assistance concept, not a substitute for professional diagnosis.

This context is intentionally included in the project's framing — it demonstrates an understanding of the gap between a working prototype and a real-world deployable medical AI system, which is a stronger signal of maturity than presenting the tool as "hospital-ready."

---

## One-Line Summary (for resume/LinkedIn)

> Built an end-to-end deep learning pipeline that detects diabetic retinopathy from retinal fundus images using transfer learning (ResNet50), achieving [X]% accuracy and [Y] F1-score on held-out test data, with Grad-CAM-based interpretability and a live Streamlit demo for real-time predictions.

## Three-Line Project Description

Built an end-to-end deep learning pipeline that classifies retinal fundus images as DR (Diabetic Retinopathy) or No DR using transfer learning with ResNet50. Achieved [X]% accuracy and [Y] F1-score on held-out test data, with Grad-CAM heatmaps added for model interpretability. Deployed as an interactive Streamlit web app allowing real-time image upload and prediction with visual explanation.
