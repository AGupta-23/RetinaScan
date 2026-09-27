"""Load the ResNet50 checkpoint once per Streamlit session."""

import os
import tempfile
import urllib.request

import streamlit as st
import torch

from config import MODEL_PATH, MODEL_URL
from src.model import build_model


@st.cache_resource
def load_model():
    """
    Build architecture, load checkpoint weights, set to eval mode.

    Resolution order for the checkpoint file:
      1. Local path (models/best_model.pth) -- used during local development.
      2. GitHub Release download            -- used on Streamlit Community
         Cloud (or any env) where the .pth file is not in the git repo
         (it is gitignored). Downloaded once to a temp dir and cached there
         for the life of the running instance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device=device)

    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        cache_dir = os.path.join(tempfile.gettempdir(), "retinascan")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "best_model.pth")
        if not os.path.exists(model_path):
            req = urllib.request.Request(
                MODEL_URL,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req) as response, open(model_path, "wb") as out_file:
                out_file.write(response.read())

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Handle both raw state_dict saves and wrapped saves ({"model_state_dict": ...})
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model, device
