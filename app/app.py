"""
app.py -- RetinaScan Streamlit Demo (Phase 10)

Local/cloud web app for diabetic retinopathy detection.
Loads the trained ResNet50 checkpoint, preprocesses an uploaded retinal
fundus image using the same pipeline as training, runs inference, and
displays a Grad-CAM heatmap alongside the original image.

Usage:
    streamlit run app/app.py
"""

import sys
import os
import tempfile
import urllib.request

# Allow imports from src/ regardless of where streamlit is launched from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import streamlit as st
import re
from PIL import Image

# -- src imports (never modify these files) -----------------------------------
from src.dataset import preprocess_transform, IMAGENET_MEAN, IMAGENET_STD
from src.model import build_model

# -- pytorch-grad-cam ---------------------------------------------------------
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")
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


# ---------------------------------------------------------------------------
# Cached model loader -- runs only once per Streamlit session
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

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
    """
    Build a plain-language interpretation string based on the prediction
    and how confident the model was, including a borderline-case caveat
    grounded in the actual Phase 8 test-set error analysis.
    """
    is_borderline = BORDERLINE_LOW <= prob <= BORDERLINE_HIGH

    if is_borderline:
        return (
            "This score sits in the **borderline band**, where test-set errors "
            "were concentrated. Treat it as inconclusive — not a screening outcome."
        )

    if label == "DR":
        if confidence_pct >= 90:
            return (
                "The model is **highly confident** this fundus shows diabetic "
                "retinopathy. Warm Grad-CAM regions are the pixels that drove "
                "the decision (often hemorrhages or exudates)."
            )
        return (
            "The model predicts **DR is present**, with moderate confidence. "
            "Use this as a prompt for clinical follow-up, not a final call."
        )

    if confidence_pct >= 90:
        return (
            "The model is **highly confident** it sees **no diabetic retinopathy** "
            "in the patterns it was trained to recognise."
        )
    return (
        "The model predicts **no DR**, with moderate confidence. Early or "
        "subtle cases are harder — routine screening still applies."
    )


# ---------------------------------------------------------------------------
# Grad-CAM helpers
# ---------------------------------------------------------------------------

def denormalize(tensor):
    """
    Undo ImageNet normalisation and return a float32 HxWx3 array in [0, 1]
    suitable for show_cam_on_image().
    """
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # HxWx3
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    return img.astype(np.float32)


def square_for_view(image, size=512):
    """
    Force both panes to the same pixel size. Matches training: torchvision
    Resize stretches to square, so attention lines up with the fundus.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def generate_gradcam(model, input_tensor, device, prob):
    """
    Generate a Grad-CAM heatmap overlay.

    Target layer: model.layer4[-1]  (last conv block of ResNet50)
    Target class: BinaryClassifierOutputTarget  (single-logit output)

    Returns an HxWx3 uint8 RGB overlay image.
    """
    target_layers = [model.layer4[-1]]
    # BinaryClassifierOutputTarget is correct for single-logit output;
    # do NOT use ClassifierOutputTarget (that is for multi-class softmax).
    targets = [BinaryClassifierOutputTarget(None)]

    input_on_device = input_tensor.to(device)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_on_device, targets=targets)

    grayscale_cam = grayscale_cam[0]       # HxW float in [0, 1]
    rgb_image = denormalize(input_tensor)  # HxWx3 float in [0, 1]

    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return overlay                         # HxWx3 uint8


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,500&family=Figtree:wght@400;500;600;700&display=swap');

        :root {
            --ink: #0A0C0F;
            --panel: #12161C;
            --line: rgba(212, 160, 84, 0.18);
            --gold: #D4A054;
            --gold-soft: rgba(212, 160, 84, 0.12);
            --paper: #E6E2D8;
            --mute: #8F887A;
            --dr: #E25B4A;
            --ok: #3CB89A;
            --warn: #E0A03A;
        }

        html, body, [class*="css"] { font-family: "Figtree", sans-serif; }

        .stApp {
            background:
                radial-gradient(1200px 500px at 12% -10%, rgba(212, 160, 84, 0.08), transparent 55%),
                radial-gradient(900px 600px at 100% 0%, rgba(60, 184, 154, 0.05), transparent 50%),
                var(--ink);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { right: 1rem; }
        footer { visibility: hidden; height: 0; }
        #MainMenu { visibility: hidden; }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #10141A 0%, #0B0E12 100%);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] > div:first-child { background: transparent; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            font-family: "Cormorant Garamond", serif;
            letter-spacing: 0.02em;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            padding: 0.35rem 0 1.1rem;
        }
        .brand-mark {
            width: 44px;
            height: 44px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .brand-mark svg {
            width: 44px;
            height: 44px;
            display: block;
            overflow: visible;
            filter: drop-shadow(0 0 12px rgba(212,160,84,0.28));
        }
        .brand .title {
            font-family: "Cormorant Garamond", serif;
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--paper);
            margin: 0;
            line-height: 1;
        }
        .brand span {
            display: block;
            font-family: "Figtree", sans-serif;
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--mute);
            margin-top: 0.28rem;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.55rem;
            margin: 0.4rem 0 1rem;
        }
        .stat {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--line);
            border-radius: 0.7rem;
            padding: 0.7rem 0.75rem 0.6rem;
        }
        .stat b {
            display: block;
            font-size: 1.05rem;
            color: var(--paper);
            font-weight: 600;
        }
        .stat i {
            font-style: normal;
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--mute);
        }

        .side-note {
            font-size: 0.78rem;
            line-height: 1.45;
            color: var(--mute);
            border-top: 1px solid var(--line);
            padding-top: 0.9rem;
            margin-top: 0.4rem;
        }

        .empty {
            text-align: center;
            padding: 2.4rem 1rem 2.6rem;
        }
        .iris {
            width: 148px;
            height: 148px;
            margin: 0 auto 1.35rem;
            border-radius: 50%;
            position: relative;
            background:
                radial-gradient(circle at 50% 50%, transparent 18%, rgba(212,160,84,0.08) 19%, transparent 20%),
                radial-gradient(circle at 38% 34%, #e8c98a 0 8%, #b56a22 18%, #7a2e24 36%, #2a1014 52%, #07080a 68%);
            box-shadow:
                0 0 0 10px rgba(212,160,84,0.05),
                0 0 0 18px rgba(212,160,84,0.04),
                0 20px 50px rgba(0,0,0,0.45);
            animation: pulse 4.8s ease-in-out infinite;
        }
        .iris::after {
            content: "";
            position: absolute;
            inset: 22%;
            border-radius: 50%;
            background: radial-gradient(circle at 40% 35%, #1c1210, #050506 70%);
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 10px rgba(212,160,84,0.05), 0 0 0 18px rgba(212,160,84,0.04), 0 20px 50px rgba(0,0,0,0.45); }
            50% { box-shadow: 0 0 0 12px rgba(212,160,84,0.09), 0 0 0 24px rgba(212,160,84,0.04), 0 20px 50px rgba(0,0,0,0.45); }
        }
        .empty .title {
            font-family: "Cormorant Garamond", serif;
            font-size: 2.05rem;
            font-weight: 600;
            color: var(--paper);
            margin: 0 0 0.45rem;
        }
        .empty p {
            color: var(--mute);
            max-width: 420px;
            margin: 0 auto;
            font-size: 0.95rem;
        }

        .report {
            margin-top: 1rem;
            border-radius: 1.05rem;
            padding: 1.15rem 1.3rem 1.2rem;
            border: 1px solid var(--line);
        }
        .report-dr { background: linear-gradient(135deg, rgba(226,91,74,0.16), rgba(18,22,28,0.4)); border-color: rgba(226,91,74,0.4); }
        .report-ok { background: linear-gradient(135deg, rgba(60,184,154,0.14), rgba(18,22,28,0.4)); border-color: rgba(60,184,154,0.38); }
        .report-warn { background: linear-gradient(135deg, rgba(224,160,58,0.16), rgba(18,22,28,0.4)); border-color: rgba(224,160,58,0.42); }

        .kicker {
            font-size: 0.68rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--mute);
            margin: 0 0 0.35rem;
        }
        .verdict {
            font-family: "Cormorant Garamond", serif;
            font-size: 2.15rem;
            font-weight: 700;
            line-height: 1;
            margin: 0 0 0.85rem;
            color: var(--paper);
        }

        .gauge-wrap { margin: 0.2rem 0 0.85rem; }
        .gauge-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--mute);
            margin-bottom: 0.35rem;
        }
        .gauge {
            position: relative;
            height: 10px;
            border-radius: 99px;
            background: linear-gradient(90deg, #3CB89A 0%, #c9b07a 50%, #E25B4A 100%);
            opacity: 0.9;
        }
        .gauge-zone {
            position: absolute;
            top: -3px;
            height: 16px;
            background: rgba(255,255,255,0.12);
            border-left: 1px dashed rgba(255,255,255,0.25);
            border-right: 1px dashed rgba(255,255,255,0.25);
        }
        .needle {
            position: absolute;
            top: -7px;
            width: 4px;
            height: 24px;
            margin-left: -2px;
            border-radius: 99px;
            background: #fff;
            box-shadow: 0 0 10px rgba(255,255,255,0.55);
        }
        .thresh {
            position: absolute;
            top: -5px;
            width: 1px;
            height: 20px;
            background: rgba(255,255,255,0.55);
        }
        .read { color: var(--paper); font-size: 0.95rem; line-height: 1.55; margin: 0; }
        .read strong { color: #fff; }

        .cam-legend {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0.15rem 0 0.9rem;
            color: var(--mute);
            font-size: 0.82rem;
        }
        .cam-bar {
            width: 92px;
            height: 8px;
            border-radius: 99px;
            background: linear-gradient(90deg, #1d4ed8, #22c55e, #eab308, #ef4444);
            flex-shrink: 0;
        }

        [data-testid="stFileUploader"] { margin-bottom: 0.2rem; }
        [data-testid="stFileUploaderDropzone"] {
            background: var(--gold-soft) !important;
            border: 1px dashed rgba(212,160,84,0.45) !important;
            border-radius: 0.9rem !important;
        }

        .st-key-workbench [data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }
        .st-key-workbench [data-testid="stHorizontalBlock"] > div {
            flex: 1 1 0 !important;
            min-width: 0;
        }
        .st-key-workbench [data-testid="stHorizontalBlock"] [data-testid="stImage"] {
            width: 100%;
        }
        .st-key-workbench [data-testid="stHorizontalBlock"] [data-testid="stImage"] img {
            width: 100% !important;
            height: auto !important;
            aspect-ratio: 1 / 1;
            object-fit: contain;
            background: #07080a;
            border-radius: 0.85rem;
            border: 1px solid var(--line);
        }

        .st-key-workbench {
            border: 1px solid var(--line);
            border-radius: 1.15rem;
            background: linear-gradient(180deg, rgba(18,22,28,0.92), rgba(10,12,15,0.75));
            padding: 1.15rem 1.25rem 1.3rem;
        }

        details {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--line);
            border-radius: 0.75rem;
            padding: 0.15rem 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def probability_gauge(prob):
    """Single visual for P(DR): zone, threshold, and needle. No duplicate bars."""
    left = BORDERLINE_LOW * 100
    width = (BORDERLINE_HIGH - BORDERLINE_LOW) * 100
    needle = prob * 100
    thresh = THRESHOLD * 100
    return f"""
        <div class="gauge-wrap">
            <div class="gauge-labels">
                <span>No DR</span>
                <span>P(DR) = {prob:.3f}</span>
                <span>DR</span>
            </div>
            <div class="gauge">
                <div class="gauge-zone" style="left:{left}%; width:{width}%;"></div>
                <div class="thresh" style="left:{thresh}%;"></div>
                <div class="needle" style="left:{needle}%;"></div>
            </div>
        </div>
    """


def render_sidebar(device):
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <div class="brand-mark" role="img" aria-label="Eye">
                    <svg viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3 22C8.5 11.5 15.5 7 22 7s13.5 4.5 19 15c-5.5 10.5-12.5 15-19 15S8.5 32.5 3 22Z"
                              fill="#F3EDE0" stroke="#D4A054" stroke-width="1.2"/>
                        <circle cx="22" cy="22" r="8.4" fill="#B56A22"/>
                        <circle cx="22" cy="22" r="8.4" fill="url(#iris)"/>
                        <circle cx="22" cy="22" r="8.4" stroke="#7A3A14" stroke-width="0.7" opacity="0.7"/>
                        <circle cx="22" cy="22" r="3.6" fill="#0C0D10"/>
                        <circle cx="19.6" cy="19.4" r="1.45" fill="#F7E7C6"/>
                        <defs>
                            <radialGradient id="iris" cx="38%" cy="34%" r="70%">
                                <stop offset="0%" stop-color="#E8C98A"/>
                                <stop offset="45%" stop-color="#C17A28"/>
                                <stop offset="100%" stop-color="#6B2A1C"/>
                            </radialGradient>
                        </defs>
                    </svg>
                </div>
                <div>
                    <div class="title">RetinaScan</div>
                    <span>Fundus screening lab</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="stat-grid">
                <div class="stat"><i>Accuracy</i><b>96.18%</b></div>
                <div class="stat"><i>F1</i><b>0.962</b></div>
                <div class="stat"><i>Precision</i><b>0.964</b></div>
                <div class="stat"><i>Recall</i><b>0.961</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Held-out 550-image split · running on **{device}**")

        with st.expander("How the model reads an image"):
            st.markdown(
                """
APTOS 2019 fundus photos, 5-class grades collapsed to **DR / No DR**.
ResNet50 (ImageNet), `layer4` fine-tuned, single logit + BCE.

**Honest limits:** not a medical device; one public dataset (domain shift
is real); a few errors came from attention on the optic disc rather than
lesions.
                """
            )

        st.markdown(
            '<p class="side-note">Portfolio proof-of-concept. Not clinically '
            "validated — do not use for diagnosis.</p>",
            unsafe_allow_html=True,
        )


def render_empty_state():
    st.markdown(
        """
        <div class="empty">
            <div class="iris"></div>
            <div class="title">The disc is waiting</div>
            <p>Upload a fundus photograph. The reading is three things:
            the retina, where the network looked, and a single probability.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report(prob, label, confidence_pct, is_borderline):
    if is_borderline:
        kind, verdict = "report-warn", "Inconclusive"
    elif label == "DR":
        kind, verdict = "report-dr", "Diabetic retinopathy"
    else:
        kind, verdict = "report-ok", "No diabetic retinopathy"

    reading = re.sub(
        r"\*\*(.+?)\*\*",
        r"<strong>\1</strong>",
        interpret_result(prob, label, confidence_pct),
    )

    st.markdown(
        f"""
        <div class="report {kind}">
            <p class="kicker">Finding · {confidence_pct:.1f}% toward this class</p>
            <p class="verdict">{verdict}</p>
            {probability_gauge(prob)}
            <p class="read">{reading}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="RetinaScan",
        page_icon="◉",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    with st.spinner("Warming the checkpoint…"):
        model, device = load_model()

    render_sidebar(device)

    with st.container(key="workbench"):
        uploaded_file = st.file_uploader(
            "Fundus image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="JPG or PNG retinal fundus photograph",
        )

        if uploaded_file is None:
            render_empty_state()
            return

        pil_image = Image.open(uploaded_file).convert("RGB")
        input_tensor = preprocess(pil_image)
        prob, label, confidence_pct = run_inference(model, input_tensor, device)
        is_borderline = BORDERLINE_LOW <= prob <= BORDERLINE_HIGH

        st.markdown(
            """
            <div class="cam-legend">
                <div class="cam-bar"></div>
                <span>Grad-CAM · cool = little influence, warm = drove the prediction</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner("Painting attention…"):
            overlay = generate_gradcam(model, input_tensor, device, prob)

        col_orig, col_cam = st.columns(2, gap="medium")
        with col_orig:
            st.image(square_for_view(pil_image), caption="Fundus", width="stretch")
        with col_cam:
            st.image(square_for_view(overlay), caption="Attention", width="stretch")

        render_report(prob, label, confidence_pct, is_borderline)


if __name__ == "__main__":
    main()
