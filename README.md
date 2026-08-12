# RetinaScan — Explainable Deep Learning for Diabetic Retinopathy Detection

RetinaScan is an end-to-end deep learning pipeline that classifies retinal fundus images as **DR (Diabetic Retinopathy present)** or **No DR**, using transfer learning on a pretrained ResNet50. The project covers the complete workflow — data collection, cleaning, preprocessing, exploratory analysis, model training, evaluation, Grad-CAM explainability, and deployment as a live interactive demo.

A user uploads a retinal (fundus) eye image, and the system returns a prediction ("DR" or "No DR") with a confidence score, along with a Grad-CAM heatmap showing which regions of the eye the model focused on to make that decision.

> This is a student/portfolio-level proof-of-concept, not a certified medical device. It demonstrates the feasibility and workflow of AI-assisted DR screening, not a deployment-ready clinical tool.

---

## Problem Statement

Diabetic Retinopathy is a diabetes complication that damages blood vessels in the retina and is a leading cause of preventable blindness. Early detection through retinal screening is effective, but manual screening requires trained ophthalmologists, who are in short supply relative to the diabetic population — particularly in countries like India. This project explores whether a deep learning model can automatically flag DR presence from a retinal image, with visual explanations to support (not replace) clinical judgment.

---

## Dataset

- **Source:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) (public, Kaggle)
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

**Workflow note:** Development happens on a storage-limited laptop, so all heavy compute (training, GPU work) runs on Kaggle Notebooks. The laptop only holds code, which is pushed to GitHub and pulled into Kaggle at runtime. Datasets and large model weights never touch the laptop — only the final trained model file and small output artifacts (metrics, plots, sample heatmaps) are downloaded locally to power the demo app.

---

## Machine Learning Approach

**Why deep learning:** A CNN learns hierarchical visual features directly from raw pixels rather than relying on hand-engineered features. Early layers learn edges and color gradients, middle layers combine these into shapes (vessel structures, spots), and deep layers learn complex, task-specific patterns associated with DR (hemorrhages, exudates, vessel damage) — all learned automatically from labeled data via backpropagation.

**Why transfer learning:** Training a CNN from scratch requires far more data and compute than is available here. Instead, ResNet50 — pretrained on ImageNet — is used as a starting point. Its early layers (general visual features) are frozen, while the last few layers are unfrozen and fine-tuned specifically on retinal images.

**Training details:**
- Loss function: Binary Cross-Entropy
- Optimizer: Adam
- Epochs: modest range (10–20), monitored for overfitting via train/validation loss and accuracy curves
- Best checkpoint saved based on validation performance (`best_model.pth`)

---

## Evaluation

The trained model is evaluated on a held-out test set using accuracy, F1-score, precision, recall, and a confusion matrix. Misclassifications are analyzed for patterns (e.g., borderline mild DR cases being harder to classify), feeding into an honest discussion of the model's limitations.

---

## Explainability — Grad-CAM

A trained deep learning model is normally a "black box" — it outputs a prediction with no visible reasoning, which is a serious trust problem in any medical-adjacent context. **Grad-CAM** addresses this by generating a heatmap over the input image, highlighting which pixel regions most influenced the model's prediction. Heatmaps are generated for sample test images and overlaid on the originals, turning the system from a black-box classifier into an interpretable one.

---

## Deployment

The final trained model is downloaded from Kaggle and integrated into a **Streamlit web app**:

1. User uploads a retinal fundus image
2. App preprocesses the image (resize, normalize) to match training conditions
3. Model returns a prediction (DR / No DR) with a confidence score
4. Grad-CAM heatmap is displayed alongside the original image

---

## Repo Structure

```
retinascan/
├── data/          # placeholder only — datasets live on Kaggle, never here
├── notebooks/     # Kaggle notebook code, synced via git
├── src/           # reusable Python modules (preprocessing, model, training, gradcam)
├── models/        # downloaded best_model.pth
├── app/           # Streamlit app code
├── outputs/       # metrics CSVs, plots, sample Grad-CAM images
├── README.md
└── requirements.txt
```

---

## How to Run Locally

```bash
git clone <your-repo-url>
cd retinascan
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Results

| Metric | Value |
|---|---|
| Accuracy | _TBD_ |
| F1-score | _TBD_ |
| Precision | _TBD_ |
| Recall | _TBD_ |

*(To be filled in after training and evaluation.)*

---

## Limitations

This project is a proof-of-concept, not a clinically deployable tool:

- **No regulatory approval:** Real hospital deployment would require CDSCO (India) approval as the tool falls under Software as a Medical Device (SaMD) classification.
- **Domain shift risk:** The model is trained on APTOS 2019 images collected under specific conditions; real hospital images (different cameras, lighting, patient populations) may differ enough to hurt performance.
- **No clinical validation:** Metrics are computed on a held-out split of the same dataset, not validated against real patient outcomes or reviewed by ophthalmologists.
- **Not a diagnostic replacement:** The system is designed as a screening-assistance concept, not a substitute for professional diagnosis.

---

## Summary

Built an end-to-end deep learning pipeline that detects diabetic retinopathy from retinal fundus images using transfer learning (ResNet50), with Grad-CAM-based interpretability and a live Streamlit demo for real-time predictions.
