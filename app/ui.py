"""Streamlit layout: styles, sidebar, empty state, finding card, workbench."""

import base64
import io

import streamlit as st
from PIL import Image

from config import IRIS_ICON_PATH, STYLES_PATH
from explain import generate_gradcam, pad_for_view
from inference import is_borderline, preprocess, run_inference


@st.cache_data
def iris_icon_data_uri(size=128):
    """Human-iris photo as a data URI (sidebar mark and empty state)."""
    img = Image.open(IRIS_ICON_PATH).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def inject_css():
    with open(STYLES_PATH, encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


def how_the_model_html():
    return (
        '<div class="how">'
        '<p class="how-lead">The photo is prepared the same way as in training. Then the model '
        "gives a yes, no, or not-sure, and shows which parts of the eye it used.</p>"
        '<div class="how-step"><div class="how-n">1</div><div>'
        "<h4>Dataset</h4>"
        "<p>APTOS 2019 fundus photographs. Kaggle’s five severity grades were collapsed into "
        "<b>DR</b> (any retinopathy) vs <b>No DR</b>, so this demo is a binary screen, not a 0–4 scale.</p>"
        "</div></div>"
        '<div class="how-step"><div class="how-n">2</div><div>'
        "<h4>Same preprocessing</h4>"
        "<p>The file is opened as RGB, resized to <b>224×224</b>, and normalised with ImageNet mean and "
        "standard deviation — the exact transform used while training ResNet50.</p>"
        "</div></div>"
        '<div class="how-step"><div class="how-n">3</div><div>'
        "<h4>The model</h4>"
        "<p>A ResNet50 network, already trained on everyday photos, was then trained on these eye photos "
        "to answer only: diabetic retinopathy, or not.</p>"
        "</div></div>"
        '<div class="how-step"><div class="how-n">4</div><div>'
        "<h4>Yes, no, or not sure</h4>"
        "<p>A score from 0 to 100. At 50 or above we say yes. Scores in the middle "
        "(about 35 to 65) are treated as not sure, because that is where mistakes clustered in testing.</p>"
        "</div></div>"
        '<div class="how-step"><div class="how-n">5</div><div>'
        "<h4>What stood out</h4>"
        "<p>A colour overlay is drawn on the same photo. Warmer colour means that part "
        "of the eye mattered more to the model. It is not a doctor’s drawing of disease.</p>"
        "</div></div>"
        '<div class="how-limits"><b>Limits.</b> Not a medical device. Trained on one public dataset, so '
        "new cameras or populations can shift results. A few errors came from attention on the optic "
        "disc rather than lesions. Held-out test: 550 images, 96.18% accuracy, F1 0.962.</div>"
        "</div>"
    )


def render_sidebar(device):
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand">
                <div class="brand-mark" role="img" aria-label="Human iris">
                    <img src="{iris_icon_data_uri()}" alt="Human iris"/>
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
            st.markdown(how_the_model_html(), unsafe_allow_html=True)

        st.markdown(
            '<p class="side-note">Portfolio proof-of-concept. Not clinically '
            "validated — do not use for diagnosis.</p>",
            unsafe_allow_html=True,
        )


def render_empty_state():
    st.markdown(
        f"""
        <div class="empty">
            <div class="iris" role="img" aria-label="Human iris">
                <img src="{iris_icon_data_uri(296)}" alt="Human iris"/>
            </div>
            <div class="title">Add a photo of an eye</div>
            <p>Upload a picture of the back of the eye. You will see DR or No DR, then a percentage, then a heatmap.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workbench(model, device):
    with st.container(key="workbench"):
        uploaded_file = st.file_uploader(
            "Photo of the eye",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="A photo of the back of the eye",
        )

        if uploaded_file is None:
            render_empty_state()
            return

        pil_image = Image.open(uploaded_file).convert("RGB")
        input_tensor = preprocess(pil_image)
        prob, label, confidence_pct = run_inference(model, input_tensor, device)
        borderline = is_borderline(prob)

        with st.spinner("Making the heatmap…"):
            overlay = generate_gradcam(model, input_tensor, device, prob)

        # --- side-by-side, matched-size images ---
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown('<p class="pane-label">Your photo</p>', unsafe_allow_html=True)
            st.image(pad_for_view(pil_image, size=480, inset=0.02), width="stretch")
        with col2:
            st.markdown('<p class="pane-label">Heatmap</p>', unsafe_allow_html=True)
            st.image(pad_for_view(overlay, size=480, inset=0.02), width="stretch")

        st.markdown(
            '<div class="heat-key">'
            '<div class="cam-bar"></div>'
            "<p><b>Blue</b> little attention &nbsp;·&nbsp; "
            "<b>Green</b> some &nbsp;·&nbsp; "
            "<b>Yellow</b> more &nbsp;·&nbsp; "
            "<b>Red</b> most attention</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # --- big centered result card ---
        if borderline:
            kind, verdict = "report-warn", "Not sure"
            icon = (
                '<svg viewBox="0 0 24 24" fill="none" stroke="var(--warn)" stroke-width="2">'
                '<circle cx="12" cy="12" r="10"/>'
                '<path d="M12 8v4M12 16h.01" stroke-linecap="round"/>'
                "</svg>"
            )
        elif label == "DR":
            kind, verdict = "report-dr", "DR detected"
            icon = (
                '<svg viewBox="0 0 24 24" fill="none" stroke="var(--dr)" stroke-width="2">'
                '<path d="M12 2 1 21h22L12 2z" stroke-linejoin="round"/>'
                '<path d="M12 9v5M12 17h.01" stroke-linecap="round"/>'
                "</svg>"
            )
        else:
            kind, verdict = "report-ok", "No DR detected"
            icon = (
                '<svg viewBox="0 0 24 24" fill="none" stroke="var(--ok)" stroke-width="2">'
                '<circle cx="12" cy="12" r="10"/>'
                '<path d="m8 12 3 3 5-6" stroke-linecap="round" stroke-linejoin="round"/>'
                "</svg>"
            )

        extra = (
            '<p class="read">The score is too close to the middle. Do not treat this as a yes or a no.</p>'
            if borderline
            else ""
        )

        st.markdown(
            f'<div class="result-card {kind}">'
            f'<div class="result-icon">{icon}</div>'
            f'<p class="verdict">{verdict}</p>'
            f'<p class="confidence">{confidence_pct:.0f}<span>%</span></p>'
            f"{extra}"
            "</div>",
            unsafe_allow_html=True,
        )