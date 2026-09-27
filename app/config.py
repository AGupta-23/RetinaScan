"""Paths and decision thresholds for the Streamlit demo."""

import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MODEL_PATH = os.path.join(REPO_ROOT, "models", "best_model.pth")
IRIS_ICON_PATH = os.path.join(APP_DIR, "assets", "iris.png")
STYLES_PATH = os.path.join(APP_DIR, "styles.css")

THRESHOLD = 0.5

# Borderline zone around the decision threshold. Phase 8 testing found that
# 3 of 11 false negatives had raw probabilities in the 0.40-0.48 range --
# i.e. cases the model got wrong while being only moderately "confident"
# about it. Predictions landing in this band get an extra caution note.
BORDERLINE_LOW = 0.35
BORDERLINE_HIGH = 0.65

# GitHub Release asset where best_model.pth is stored, used as a fallback
# when the local checkpoint isn't present (e.g. on Streamlit Community Cloud).
MODEL_URL = "https://github.com/AGupta-23/RetinaScan/releases/download/v1.0-model/best_model.pth"
