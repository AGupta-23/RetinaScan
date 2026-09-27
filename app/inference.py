"""Preprocess a fundus photo, score DR, and phrase the finding."""

import numpy as np
import torch

from config import BORDERLINE_HIGH, BORDERLINE_LOW, THRESHOLD
from src.dataset import preprocess_transform


def preprocess(pil_image):
    """
    Preprocess a PIL RGB image using the exact same pipeline as training.

    Pipeline mirrors dataset.py:
        PIL.Image -> np.array -> preprocess_transform
        preprocess_transform starts with ToPILImage(), so it expects
        a numpy array or tensor, NOT a raw PIL image.
    """
    img_array = np.array(pil_image)           # HxWx3, uint8, RGB
    tensor = preprocess_transform(img_array)  # CxHxW, float32, normalised
    return tensor.unsqueeze(0)                # 1xCxHxW batch dim


def run_inference(model, input_tensor, device):
    """
    Run forward pass and return (probability, label, confidence_pct).

    The model outputs a single raw logit trained with BCEWithLogitsLoss, so
    sigmoid must be applied manually -- it is NOT built into the model.
    """
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logit = model(input_tensor)           # shape: (1, 1)
    prob = torch.sigmoid(logit).item()        # scalar in [0, 1]

    label = "DR" if prob >= THRESHOLD else "No DR"
    confidence_pct = prob * 100 if prob >= THRESHOLD else (1 - prob) * 100
    return prob, label, confidence_pct


def interpret_result(prob, label, confidence_pct):
    """One plain sentence for the person looking at the screen."""
    if BORDERLINE_LOW <= prob <= BORDERLINE_HIGH:
        return (
            "The answer is **not clear**. The score is too close to the middle. "
            "Do not treat this as a yes or a no."
        )

    if label == "DR":
        if confidence_pct >= 90:
            return "This eye **shows diabetic retinopathy**. The model is very sure."
        return (
            "This eye **likely shows diabetic retinopathy**. The model is only "
            "somewhat sure — a doctor should still look at it."
        )

    if confidence_pct >= 90:
        return "This eye **does not show diabetic retinopathy**. The model is very sure."
    return (
        "This eye **does not look like diabetic retinopathy**, but the model "
        "is only somewhat sure."
    )


def is_borderline(prob):
    return BORDERLINE_LOW <= prob <= BORDERLINE_HIGH