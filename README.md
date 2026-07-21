# RetinaVision AI — Progress Tracker

**Currently On:** Phase 0, Task 0.1
**Last Updated:** <fill in date each time you work on this>
**Kaggle Notebooks Created So Far:** (add links as you create them)

---

## Phase 0: Project Setup
- [ ] 0.1 Create local project folder structure
- [ ] 0.2 Set up git + push skeleton to GitHub
- [ ] 0.3 Create Kaggle account + verify phone (for GPU access)
- [ ] 0.4 Locate/bookmark APTOS 2019 dataset on Kaggle
- [ ] 0.5 Set up local Python environment + requirements.txt
- [ ] 0.6 Write initial README.md

## Phase 1: Kaggle Setup + EDA
- [ ] 1.1 Create first Kaggle Notebook (retinavision-01-eda)
- [ ] 1.2 Enable GPU + internet, check quota
- [ ] 1.3 Load and inspect train.csv
- [ ] 1.4 Analyze class distribution (note % imbalance here: _____)
- [ ] 1.5 Visually inspect sample images per class
- [ ] 1.6 Check image size/aspect ratio variability
- [ ] 1.7 Save EDA outputs + download figures locally + save Kaggle notebook version

## Phase 2: Preprocessing Pipeline
- [ ] 2.1 Write preprocessing.py locally (crop, resize, enhance)
- [ ] 2.2 Pull into new Kaggle notebook (retinavision-02-preprocessing)
- [ ] 2.3 Visually validate preprocessing on samples
- [ ] 2.4 Run preprocessing on full dataset, save as Kaggle output dataset
- [ ] 2.5 Set up augmentation pipeline (albumentations)
- [ ] 2.6 Handle class imbalance (method chosen: _____)
- [ ] 2.7 Create stratified train/val/test splits
- [ ] 2.8 Build and test Dataset/DataLoader class

## Phase 3: Model Architecture & Training
- [ ] 3.1 Choose backbone (chosen: _____)
- [ ] 3.2 Modify classification head (approach: classification / ordinal regression — chosen: _____)
- [ ] 3.3 Set up loss function
- [ ] 3.4 Set up optimizer + LR schedule
- [ ] 3.5 Create training Kaggle notebook (retinavision-03-training)
- [ ] 3.6 Stage 1 training (frozen backbone) — result: _____
- [ ] 3.7 Stage 2 training (fine-tuning) — result: _____
- [ ] 3.8 Add checkpointing + early stopping
- [ ] 3.9 Run full training, save notebook version
- [ ] 3.10 Download final best model to local models/ folder

## Phase 4: Explainable AI — Grad-CAM
- [ ] 4.1 Write gradcam.py locally
- [ ] 4.2 Test in Kaggle notebook
- [ ] 4.3 Generate heatmap overlays
- [ ] 4.4 Clinically sanity-check results — notes: _____
- [ ] 4.5 Save + download sample outputs

## Phase 5: Evaluation
- [ ] 5.1 Run inference on test set
- [ ] 5.2 Compute Accuracy/Precision/Recall/F1/ROC-AUC
- [ ] 5.3 Compute Quadratic Weighted Kappa — score: _____
- [ ] 5.4 Build + analyze confusion matrix
- [ ] 5.5 (Optional) Cross-dataset validation — done? Y/N, result: _____
- [ ] 5.6 Download all metrics locally

## Phase 6: Local Demo App
- [ ] 6.1 Choose interface (Streamlit / Gradio): _____
- [ ] 6.2 Build upload + inference flow
- [ ] 6.3 Add polish (class descriptions, disclaimer)
- [ ] 6.4 Test end to end locally
- [ ] 6.5 (Optional) Deploy publicly — link: _____

## Phase 7: Documentation & Final Report
- [ ] 7.1 Finalize README.md
- [ ] 7.2 Write limitations section
- [ ] 7.3 Compile final report (if applicable)
- [ ] 7.4 Final GitHub cleanup

---

## Notes / Blockers Log
(Use this space to jot down anything you got stuck on, so future-you has context)

-