"""
RetinaScan Streamlit demo (Phase 10).

Local/cloud web app for diabetic retinopathy detection: load the trained
ResNet50 checkpoint, score an uploaded fundus photograph, and show Grad-CAM.

Usage:
    streamlit run app/app.py
"""

import streamlit as st

import config  # noqa: F401  — puts the repo root on sys.path
from loader import load_model
from ui import inject_css, render_sidebar, render_workbench


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
    render_workbench(model, device)


if __name__ == "__main__":
    main()
