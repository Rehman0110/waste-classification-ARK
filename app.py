# =============================================================
# Smart Waste Segregation – FINAL APP (UPLOAD + CAMERA TOGGLE)
# 2 Classes: Organic, Recyclable
# =============================================================

import os
import sys
import json
from datetime import datetime
import requests

import streamlit as st
import numpy as np
from PIL import Image

# ── PATH SETUP ────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.voice_output import announce_prediction, get_suggestion


# ── CONSTANTS ────────────────────────────────────────────────
CLASS_NAMES = ["Organic", "Recyclable"]

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

CLASS_COLOURS = {
    "Recyclable": "#2196F3",
    "Organic": "#4CAF50"
}

HISTORY_FILE = os.path.join(ROOT, "static", "history.json")
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)


# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config("Waste Classifier", "♻️", layout="wide")


# ── SESSION STATE ─────────────────────────────────────────────
def init_state():
    # Check if API is online
    try:
        api_base = API_URL.replace("/predict", "")
        requests.get(f"{api_base}/", timeout=2)
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Waste Segregation API is offline. Please run 'python api.py' first.")
        
    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

    if "last_run" not in st.session_state:
        st.session_state.last_run = None

init_state()


# ── HISTORY ───────────────────────────────────────────────────
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except:
        return []


def save_prediction(cls, conf):
    history = load_history()

    entry = {
        "class": cls,
        "confidence": float(conf),
        "timestamp": datetime.now().isoformat()
    }

    # prevent duplicates
    if history:
        last = history[-1]
        if last["class"] == entry["class"] and abs(last["confidence"] - entry["confidence"]) < 1e-4:
            return

    history.append(entry)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ── UI ────────────────────────────────────────────────────────
st.title("♻️ Smart Waste Segregation System")

tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Camera"])


# =============================================================
# UPLOAD TAB
# =============================================================
with tab1:
    uploaded = st.file_uploader("Upload Waste Image", type=["jpg", "png", "jpeg"])


# =============================================================
# CAMERA TAB (TOGGLE ON/OFF)
# =============================================================
with tab2:

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📷 Turn ON Camera"):
            st.session_state.camera_on = True

    with col2:
        if st.button("⛔ Turn OFF Camera"):
            st.session_state.camera_on = False

    camera = None

    if st.session_state.camera_on:
        camera = st.camera_input("Capture Waste Image")
    else:
        st.info("Camera is OFF")


# =============================================================
# IMAGE SOURCE
# =============================================================
img_file = uploaded if uploaded is not None else camera


# =============================================================
# PREDICTION PIPELINE (SAFE)
# =============================================================
if img_file:

    img = Image.open(img_file).convert("RGB")
    st.image(img, use_container_width=True)

    predict = st.button("Predict", type="primary")

    if predict:

        run_id = f"{img_file.name}_{img_file.size}"

        if st.session_state.last_run == run_id:
            st.stop()

        st.session_state.last_run = run_id

        with st.spinner("Predicting..."):
            try:
                # Prepare the file for the API request
                files = {"file": (img_file.name, img_file.getvalue(), img_file.type)}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    class_name = result["prediction"]
                    confidence = result["confidence"]
                    probs = result["probabilities"]
                else:
                    st.error(f"API Error: {response.text}")
                    st.stop()
                    
            except Exception as e:
                st.error(f"Failed to connect to the prediction API: {str(e)}")
                st.stop()

        # ── SUCCESS HANDLING ──────────────────────────────────────

        save_prediction(class_name, confidence)

        info = get_suggestion(class_name)
        color = CLASS_COLOURS[class_name]

        st.markdown(f"""
        <div style="padding:20px;border-left:6px solid {color};
        background:#f5f5f5;border-radius:10px;">
        <h2>{info['icon']} {class_name}</h2>
        <p>{info['suggestion']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Confidence", f"{confidence:.2%}")
        st.progress(float(confidence))
        st.success(info["action"])

        if st.toggle("🔊 Voice Output"):
            announce_prediction(class_name, confidence)

        st.subheader("Probabilities")

        for k, v in probs.items():
            st.write(f"{get_suggestion(k)['icon']} {k}")
            st.progress(float(v))


# =============================================================
# ANALYTICS
# =============================================================
st.divider()
st.subheader("Analytics")

history = load_history()

if history:

    clean = [h for h in history if h["class"] in CLASS_NAMES]

    counts = {c: sum(1 for h in clean if h["class"] == c) for c in CLASS_NAMES}

    st.bar_chart(counts)

    st.write("Recent Predictions")

    for h in clean[-10:][::-1]:
        st.write(f"{h['class']} — {h['confidence']:.4f}")

else:
    st.info("No predictions yet")