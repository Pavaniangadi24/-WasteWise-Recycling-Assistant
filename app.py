"""
WasteWise Recycling Assistant - Streamlit Web Application.

A clean, modern, and accessible web interface for local waste classification
using MobileNetV2 transfer learning and structured recycling recommendations
via Google Gemini AI (with deterministic rule-based fallback).
"""

import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Set environment paths to avoid permission issues on Windows
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

os.environ.setdefault("KERAS_HOME", str(_project_root / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_project_root / ".matplotlib_cache"))
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import pandas as pd
from PIL import Image
import streamlit as st

from src.predict import (
    validate_image_file,
    predict_image,
    load_classifier_model,
    DEFAULT_MODEL_PATH,
)
from src.genai_assistant import (
    GenAIRecyclingAssistant,
    mask_api_key,
    DEFAULT_GEMINI_MODEL,
)

# Page configuration
st.set_page_config(
    page_title="WasteWise Recycling Assistant",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, accessible visual styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2e7d32;
        margin-bottom: 12px;
    }
    .organic-badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #a5d6a7;
    }
    .recyclable-badge {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #90caf9;
    }
    .disclaimer-box {
        background-color: #fffde7;
        border-left: 4px solid #fbc02d;
        padding: 10px 14px;
        font-size: 0.9rem;
        color: #5d4037;
        border-radius: 4px;
        margin-top: 10px;
    }
    .source-badge-gemini {
        background-color: #f3e5f5;
        color: #6a1b9a;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #ce93d8;
    }
    .source-badge-fallback {
        background-color: #fff3e0;
        color: #e65100;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #ffcc80;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Load and cache the trained MobileNetV2 model checkpoint."""
    return load_classifier_model(DEFAULT_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_assistant_instance():
    """Load and cache the GenAI Recycling Assistant instance."""
    return GenAIRecyclingAssistant()


def render_sidebar():
    """Render the sidebar navigation and operational system status."""
    st.sidebar.markdown("## ♻️ WasteWise AI")
    st.sidebar.caption("Intelligent Waste Sorting & Sustainability")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home & Overview",
            "♻️ Waste Classifier & AI Assistant",
            "📊 Dataset Explorer",
            "📈 Model Performance",
            "ℹ️ About",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ System Status")

    # Check model file status
    model_exists = Path(DEFAULT_MODEL_PATH).is_file()
    if model_exists:
        st.sidebar.success("✅ Vision Model: Loaded\n(MobileNetV2, 90.65% Acc)")
    else:
        st.sidebar.error("❌ Vision Model: Checkpoint Missing")

    # Check GenAI assistant status
    assistant = get_assistant_instance()
    is_api_ready = assistant.is_api_key_configured()
    if is_api_ready:
        st.sidebar.info(f"🌐 GenAI: Connected\n({assistant.model_name})")
    else:
        st.sidebar.warning("🛡️ GenAI: Rule-Based Fallback\n(Offline Mode Active)")

    st.sidebar.markdown("---")
    st.sidebar.caption("🔒 **Edge Privacy Notice:** Vision inference runs 100% locally. Raw image pixels are never uploaded.")

    return page


def render_home():
    """Render the Home and Overview page."""
    st.markdown('<div class="main-title">🌱 WasteWise Recycling Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Empowering responsible waste segregation through local computer vision and generative AI guidance.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
            ### Purpose & Mission
            Improper waste disposal and recycling contamination remain major global environmental challenges. 
            **WasteWise** bridges the gap between intention and action by providing **instant, accessible image classification** 
            and **actionable recycling guidance**.

            Using deep transfer learning with **MobileNetV2**, WasteWise accurately classifies common household waste into two core streams:
            - 🥬 **Organic Waste:** Biodegradable matter that safely composts into nutrient-dense soil.
            - 📦 **Recyclable Waste:** Clean, manufactured dry recyclables that can be reprocessed into new goods.
            """
        )

        st.info(
            "🔒 **Privacy First:** Image classification is performed entirely on your local machine using the optimized "
            "MobileNetV2 neural network. No photos or pixel data are uploaded to third-party servers."
        )

    with col2:
        st.markdown("### 🔄 How It Works")
        st.markdown(
            """
            1. **📷 Upload Image:**  
               Take a photo or upload an image (JPG, JPEG, PNG) of any household waste item.
            2. **⚡ Local AI Vision:**  
               The local MobileNetV2 neural network classifies the image into Organic or Recyclable in milliseconds.
            3. **🌱 Expert Guidance:**  
               Receive structured disposal instructions, composting recommendations, reuse ideas, and safety precautions via Google Gemini or the built-in offline knowledge base.
            """
        )

    st.markdown("---")
    st.markdown("### 🗑️ Understanding the Two Waste Streams")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <span class="organic-badge">🥬 Organic Waste (Class 0)</span>
                <p style="margin-top: 10px; color: #333;">
                    <strong>Typical Items:</strong> Fruit & vegetable peels, food scraps, coffee grounds, tea leaves, eggshells, garden clippings, and non-treated plant fibers.
                </p>
                <p style="color: #555; font-size: 0.95rem;">
                    <strong>Destination:</strong> Green / Brown compost bins or home compost systems.<br>
                    <strong>Environmental Impact:</strong> Composting diverts organic matter from anaerobic landfills, directly reducing emissions of methane gas (a greenhouse gas 25x more potent than CO₂).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="metric-card" style="border-left-color: #1565c0;">
                <span class="recyclable-badge">📦 Recyclable Waste (Class 1)</span>
                <p style="margin-top: 10px; color: #333;">
                    <strong>Typical Items:</strong> Rigid plastics (PETE #1, HDPE #2), clean aluminum cans, steel/tin food cans, clean cardboard, paper packaging, and glass containers.
                </p>
                <p style="color: #555; font-size: 0.95rem;">
                    <strong>Destination:</strong> Blue recycling carts or designated municipal collection bins.<br>
                    <strong>Environmental Impact:</strong> Mechanical reprocessing of aluminum saves up to 95% of the energy needed for virgin extraction; recycling paper conserves timber and water resources.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_classifier():
    """Render the Waste Classifier and AI Recycling Assistant page."""
    st.markdown('<div class="main-title">♻️ Waste Classifier & AI Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload an image to identify the waste stream and generate tailored disposal instructions.</div>',
        unsafe_allow_html=True,
    )

    # Load model
    try:
        model = get_cached_model()
    except Exception as e:
        st.error(f"Error loading classifier model: {e}")
        return

    # Image input options: File Upload or Sample Images
    input_method = st.radio(
        "Choose image source:",
        ["📁 Upload Your Own Image", "🖼️ Try a Sample Image from Test Set"],
        horizontal=True,
    )

    pil_image = None
    image_name = "Uploaded Image"

    if input_method == "📁 Upload Your Own Image":
        uploaded_file = st.file_uploader(
            "Upload waste image (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"],
            help="Images are resized to 224x224 and preprocessed locally.",
        )
        if uploaded_file is not None:
            is_valid, err_msg, loaded_img = validate_image_file(uploaded_file, filename=uploaded_file.name)
            if not is_valid:
                st.error(f"⚠️ Validation Error: {err_msg}")
            else:
                pil_image = loaded_img
                image_name = uploaded_file.name

    else:
        # Sample images from dataset test split
        sample_options = {
            "Select a sample image...": None,
            "🥬 Sample Organic 1 (O_12568.jpg)": "data/raw/DATASET/TEST/O/O_12568.jpg",
            "🥬 Sample Organic 2 (O_12569.jpg)": "data/raw/DATASET/TEST/O/O_12569.jpg",
            "📦 Sample Recyclable 1 (R_10000.jpg)": "data/raw/DATASET/TEST/R/R_10000.jpg",
            "📦 Sample Recyclable 2 (R_10001.jpg)": "data/raw/DATASET/TEST/R/R_10001.jpg",
        }
        selected_sample = st.selectbox("Select sample test item:", list(sample_options.keys()))
        sample_path = sample_options[selected_sample]

        if sample_path and Path(sample_path).is_file():
            is_valid, err_msg, loaded_img = validate_image_file(sample_path)
            if is_valid:
                pil_image = loaded_img
                image_name = Path(sample_path).name

    if pil_image is None:
        st.info("👆 Please upload an image or choose a sample image from the test set to begin.")
        return

    # Visual layout: Two columns (Image + Prediction)
    col_img, col_pred = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown(f"**Image Preview:** `{image_name}`")
        st.image(pil_image, caption=f"Original Size: {pil_image.width} × {pil_image.height} px", use_container_width=True)

    with col_pred:
        st.markdown("### 🔍 Model Classification Result")

        with st.spinner("Classifying image with MobileNetV2..."):
            pred_result = predict_image(model, pil_image)

        category = pred_result["category"]
        confidence = pred_result["confidence"]
        conf_pct = pred_result["confidence_percent"]
        raw_prob = pred_result["raw_probability"]
        latency = pred_result["inference_latency_ms"]

        # Store in session state for cross-referencing
        st.session_state["last_category"] = category
        st.session_state["last_confidence"] = confidence

        # Render Category Badge
        if category == "Organic":
            st.markdown(
                f"""
                <div class="metric-card" style="border-left-color: #2e7d32; background-color: #f1f8e9;">
                    <span class="organic-badge">🥬 Predicted: Organic Waste</span>
                    <h2 style="color: #2e7d32; margin-top: 10px; margin-bottom: 5px;">{conf_pct}% Confidence</h2>
                    <p style="color: #444; margin: 0;">Class Index: <strong>0 (Organic)</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left-color: #1565c0; background-color: #e8f0fe;">
                    <span class="recyclable-badge">📦 Predicted: Recyclable Waste</span>
                    <h2 style="color: #1565c0; margin-top: 10px; margin-bottom: 5px;">{conf_pct}% Confidence</h2>
                    <p style="color: #444; margin: 0;">Class Index: <strong>1 (Recyclable)</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Confidence Progress Bar
        st.progress(float(confidence))

        # Metrics breakdown
        m1, m2, m3 = st.columns(3)
        m1.metric("Sigmoid Score (p)", f"{raw_prob:.4f}")
        m2.metric("Inference Time", f"{latency:.1f} ms")
        m3.metric("Decision Boundary", "p ≥ 0.50")

        st.markdown(
            '<div class="disclaimer-box">⚠️ <strong>Statistical Note:</strong> Confidence reflects the model\'s mathematical '
            "probability distribution and is not a guarantee of correctness. Complex composite items or soiled materials "
            "may require manual verification against municipal sorting guidelines.</div>",
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # AI Recycling Assistant Section
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🤖 Step 2: AI Recycling Assistant")
    st.markdown(
        "Generate comprehensive, structured waste-handling guidance tailored to this item. "
        "You may optionally provide the specific item name to contextualize the recommendations."
    )

    assistant = get_assistant_instance()

    c_desc, c_btn = st.columns([3, 1], gap="medium")
    with c_desc:
        user_item_desc = st.text_input(
            "Optional item description (e.g. 'banana peel', 'aluminum soda can', 'crushed cardboard box'):",
            placeholder="Type item description here...",
            key="user_item_description_input",
        )
    with c_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        generate_clicked = st.button("🌱 Get Guidance", key="btn_generate_advice", type="primary")

    if generate_clicked:
        with st.spinner("Consulting recycling sustainability engine..."):
            advice_res = assistant.generate_recommendations(
                category=category,
                confidence=confidence,
                item_description=user_item_desc if user_item_desc.strip() else None,
            )
            st.session_state["cached_advice"] = advice_res

    # Render advice if available in session state
    if "cached_advice" in st.session_state:
        advice_res = st.session_state["cached_advice"]

        is_fallback = advice_res.get("is_fallback", False)
        source_label = advice_res.get("source", "Unknown Source")
        backend_label = advice_res.get("backend", "Engine")

        if is_fallback:
            st.markdown(
                f'<span class="source-badge-fallback">🛡️ {source_label} ({backend_label})</span>',
                unsafe_allow_html=True,
            )
            if "error" in advice_res and advice_res["error"]:
                st.caption(f"Reason: {advice_res['error']}")
        else:
            st.markdown(
                f'<span class="source-badge-gemini">🌐 {source_label} via {backend_label}</span>',
                unsafe_allow_html=True,
            )

        st.markdown(advice_res.get("formatted_text", ""))


def render_dataset_explorer():
    """Render the Dataset Explorer page."""
    st.markdown('<div class="main-title">📊 Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Detailed exploration and distribution analysis of the Sashaank Sekar Waste Classification Dataset.</div>',
        unsafe_allow_html=True,
    )

    # Top high-level metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Verified Images", "25,077", "100%")
    col2.metric("Original Training Split", "22,564", "90.0%")
    col3.metric("Isolated Testing Split", "2,513", "10.0%")
    col4.metric("Classification Classes", "2", "O & R")

    st.markdown("---")

    c_left, c_right = st.columns([1, 1], gap="large")

    with c_left:
        st.markdown("### 📁 Dataset Structure & Split Breakdown")

        df_counts = pd.DataFrame(
            [
                {"Split": "TRAIN", "Category": "Organic (O)", "Count": 12565, "Class Share": "55.68%"},
                {"Split": "TRAIN", "Category": "Recyclable (R)", "Count": 9999, "Class Share": "44.32%"},
                {"Split": "TEST (Isolated)", "Category": "Organic (O)", "Count": 1401, "Class Share": "55.75%"},
                {"Split": "TEST (Isolated)", "Category": "Recyclable (R)", "Count": 1112, "Class Share": "44.25%"},
            ]
        )
        st.dataframe(df_counts, use_container_width=True, hide_index=True)

        st.markdown("### 🔀 Stratified Training / Validation Split")
        st.markdown(
            """
            To preserve the test set's absolute isolation for final benchmarking, the **TRAIN** partition 
            (22,564 images) was partitioned via stratified 80/20 splitting (`random_state=42`):
            - **Training Set (80%):** 18,051 images (10,052 Organic, 7,999 Recyclable)
            - **Validation Set (20%):** 4,513 images (2,513 Organic, 2,000 Recyclable)
            - **Held-Out Test Set:** 2,513 images (1,401 Organic, 1,112 Recyclable)
            """
        )

        st.info(
            "🔍 **Phase 1 Inspection Finding:** The dataset contains 372 files with `.jpg` extensions that are internally "
            "encoded as palette-based PNGs (mode `'P'`). Our preprocessing pipeline explicitly decodes and converts "
            "every image to 3-channel RGB, ensuring zero corrupted or skipped samples."
        )

    with c_right:
        st.markdown("### 📈 Visual Distribution")

        # Create distribution chart dataframe
        chart_data = pd.DataFrame(
            {
                "Split": ["Train: Organic", "Train: Recyclable", "Test: Organic", "Test: Recyclable"],
                "Images": [12565, 9999, 1401, 1112],
                "Stream": ["Organic", "Recyclable", "Organic", "Recyclable"],
            }
        )
        st.bar_chart(chart_data.set_index("Split")["Images"], color="#2e7d32")

        st.caption("Figure 1: Verified class distribution across training and testing partitions.")


def render_model_performance():
    """Render the Model Performance and Benchmarking page."""
    st.markdown('<div class="main-title">📈 Model Performance & Benchmarks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Empirical evaluation of the best MobileNetV2 checkpoint evaluated on the isolated test set.</div>',
        unsafe_allow_html=True,
    )

    # Key Performance Metrics
    st.markdown("### 🎯 Benchmark Highlights (Isolated Test Set: 2,513 Images)")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Final Test Accuracy", "90.65%", "2,278 / 2,513")
    k2.metric("Test ROC-AUC", "0.9866", "High Separation")
    k3.metric("Macro F1-Score", "90.28%", "Balanced")
    k4.metric("Weighted F1-Score", "90.50%", "Weighted")
    k5.metric("Best Val Accuracy", "96.10%", "Epoch 7")

    st.markdown(
        '<div class="disclaimer-box">📌 <strong>Evaluation Rigor Note:</strong> '
        "<strong>Validation accuracy (96.10%)</strong> was achieved at Epoch 7 during classification head training on the validation split (4,513 images). "
        "<strong>Test accuracy (90.65%)</strong> was measured on the completely isolated test set (2,513 images) with zero data leakage. "
        "Validation and test results are distinct and never conflated.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Detailed Classification Report
    st.markdown("### 📋 Per-Class Performance Breakdown")
    col_t1, col_t2 = st.columns([3, 2], gap="large")

    with col_t1:
        metrics_table = pd.DataFrame(
            [
                {
                    "Class": "Organic (0)",
                    "Precision": "86.39%",
                    "Recall": "98.79%",
                    "F1-Score": "92.17%",
                    "Support": "1,401",
                },
                {
                    "Class": "Recyclable (1)",
                    "Precision": "98.13%",
                    "Recall": "80.40%",
                    "F1-Score": "88.38%",
                    "Support": "1,112",
                },
                {
                    "Class": "Macro Average",
                    "Precision": "92.26%",
                    "Recall": "89.59%",
                    "F1-Score": "90.28%",
                    "Support": "2,513",
                },
                {
                    "Class": "Weighted Average",
                    "Precision": "91.59%",
                    "Recall": "90.65%",
                    "F1-Score": "90.50%",
                    "Support": "2,513",
                },
            ]
        )
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)

        st.markdown(
            """
            **Key Metric Insights:**
            - **High Organic Recall (98.79%):** Only 17 out of 1,401 organic items were misclassified as recyclable.
            - **High Recyclable Precision (98.13%):** When the model predicts Recyclable, it is correct 98.13% of the time, protecting recycling streams from contamination.
            """
        )

    with col_t2:
        st.markdown("**Confusion Matrix Counts:**")
        cm_df = pd.DataFrame(
            [
                {"Actual Category": "Actual Organic (0)", "Pred Organic": 1384, "Pred Recyclable": 17},
                {"Actual Category": "Actual Recyclable (1)", "Pred Organic": 218, "Pred Recyclable": 894},
            ]
        )
        st.dataframe(cm_df.set_index("Actual Category"), use_container_width=True)

    st.markdown("---")

    # Render Visual Curves from outputs/
    st.markdown("### 📊 Evaluation Curves & Diagnostic Plots")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC & PR Curves", "Training History Curves"])

    with tab1:
        cm_path = _project_root / "outputs" / "confusion_matrix.png"
        if cm_path.is_file():
            st.image(str(cm_path), caption="Figure 2: Normalized Confusion Matrix on Held-Out Test Set.", use_container_width=True)
        else:
            st.warning("Confusion matrix plot not found in outputs/.")

    with tab2:
        c_roc, c_pr = st.columns(2)
        with c_roc:
            roc_path = _project_root / "outputs" / "roc_curve.png"
            if roc_path.is_file():
                st.image(str(roc_path), caption="Figure 3: ROC Curve (AUC = 0.9866).", use_container_width=True)
            else:
                st.warning("ROC curve plot not found.")
        with c_pr:
            pr_path = _project_root / "outputs" / "precision_recall_curve.png"
            if pr_path.is_file():
                st.image(str(pr_path), caption="Figure 4: Precision-Recall Curve (AP = 0.9824).", use_container_width=True)
            else:
                st.warning("PR curve plot not found.")

    with tab3:
        train_path = _project_root / "outputs" / "training_curves.png"
        if train_path.is_file():
            st.image(str(train_path), caption="Figure 5: Training and Validation Accuracy & Loss Curves.", use_container_width=True)
        else:
            st.warning("Training curves plot not found in outputs/.")


def render_about():
    """Render the About page."""
    st.markdown('<div class="main-title">ℹ️ About WasteWise</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Architecture, transfer learning methodology, tech stack, and practical limitations.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 🎯 Project Overview
        **WasteWise Recycling Assistant** is an end-to-end computer vision and Generative AI application engineered to 
        solve the real-world challenge of household waste sorting. By pairing lightweight, edge-compatible deep learning 
        with structured generative sustainability reasoning, WasteWise provides instant classification and contextual guidance.

        ### 🧠 Machine Learning Architecture
        - **Base Backbone:** `MobileNetV2` pretrained on ImageNet (frozen base of 154 layers, 2,257,984 parameters).
        - **Classification Head:** 
          - GlobalAveragePooling2D
          - BatchNormalization
          - Dropout(0.3)
          - Dense(128, activation='relu')
          - Dropout(0.2)
          - Dense(1, activation='sigmoid') (Total trainable parameters: 166,657).
        - **Input Preprocessing:** RGB 3-channel conversion, bilinear resize to 224 × 224, pixel scaling to `[-1.0, 1.0]` (`x / 127.5 - 1.0`).
        - **Saved Weights:** `models/best_mobilenetv2.keras` (11.64 MB, restored from Best Epoch 7).

        ### 🤖 Generative AI & Deterministic Fallback
        - **Gemini Assistant:** Queries Google's Gemini API (`gemini-3.8-flash` default, configurable via `.env`) with 6 structured sustainability prompts.
        - **Dual-Backend Dispatch:** Natively supports both the official `google-genai` SDK and a direct REST API client (`requests`).
        - **Privacy Architecture:** Only text metadata (category, confidence, and optional description) is transmitted when Gemini is enabled. Raw image pixels never leave the client machine.
        - **Offline Rule-Based Fallback:** An expert-curated sustainability knowledge base (`src/recycling_rules.py`) serves identical 6-section guidance when offline, ensuring 100% application uptime.

        ### 💻 Technology Stack
        - **Language & Runtime:** Python 3.12 (Windows 11)
        - **Deep Learning:** TensorFlow 2.20.0, Keras 3.12.0
        - **Web Application:** Streamlit 1.66.1
        - **Data & Image Processing:** Pillow 12.0.0, NumPy 1.26.4, Pandas 2.2.2, Scikit-Learn 1.5.1
        - **Visualization:** Matplotlib 3.11.0, Seaborn 0.13.2

        ### ⚠️ Real-World Limitations & Disclaimers
        1. **Confidence ≠ Guarantee:** Model confidence represents statistical probability from training patterns. It does not guarantee ground-truth classification.
        2. **Visual Ambiguity & Soiled Materials:** Items with hidden composite layers (e.g., plastic-lined paper coffee cups, soiled greasy pizza boxes) may visually resemble clean paper but cannot be mechanically recycled.
        3. **Local Municipal Variance:** Waste management regulations and recycling facility capabilities differ significantly between jurisdictions. Always consult local municipal sorting guidelines.
        """
    )


def main():
    """Main application dispatcher."""
    page = render_sidebar()

    if page == "🏠 Home & Overview":
        render_home()
    elif page == "♻️ Waste Classifier & AI Assistant":
        render_classifier()
    elif page == "📊 Dataset Explorer":
        render_dataset_explorer()
    elif page == "📈 Model Performance":
        render_model_performance()
    elif page == "ℹ️ About":
        render_about()


if __name__ == "__main__":
    main()
