"""
dashboard/app.py
----------------
Streamlit Real-Time Medical Anomaly Detection Dashboard

Tabs
----
1. Real-time Detection — Upload image, view Grad-CAM, confidence gauges,
   anomaly score meter.
2. Feature Analysis — PCA/t-SNE scatter, top features bar chart.
3. Model Performance — Confusion matrix, ROC, PR curves, live metrics table.
4. Batch Processing — Upload ZIP, process, download CSV results.
"""

from __future__ import annotations

import io
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Ensure imports work from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Auto-train: if model doesn't exist (fresh cloud deploy), train a quick one
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="🚀 First launch: training model (takes ~2 min)...")
def _ensure_model_exists(config_path: str) -> bool:
    """Train a fast lightweight model if none exists yet.  Runs once per deploy."""
    from src.utils import load_config
    cfg = load_config(config_path)
    model_path = Path(cfg["paths"]["model_save_path"])
    if model_path.exists():
        return True  # already trained

    from src.preprocessing import MedicalImagePreprocessor
    from src.model import MedicalCNNModel
    from sklearn.model_selection import train_test_split

    img_size = (128, 128)
    preprocessor = MedicalImagePreprocessor(image_size=img_size, augmentation_enabled=True, random_seed=42)
    images, labels = preprocessor.generate_synthetic_data(n_samples=600, image_size=img_size, random_seed=42)

    X_tr, _, y_tr, _ = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    cnn = MedicalCNNModel(
        input_shape=(128, 128, 1),
        num_classes=len(cfg["classes"]),
        class_names=cfg["classes"],
        model_save_path=model_path,
    )
    cnn.build_model()
    cnn.train(X_tr, y_tr, epochs=30, batch_size=16, validation_split=0.2,
              learning_rate=3e-4, patience=8)
    return True


# Lazy-loaded project modules (avoid heavy imports at startup)
@st.cache_resource(show_spinner="Loading pipeline components...")
def _load_pipeline(config_path: str):
    from src.utils import load_config, setup_logging
    setup_logging("WARNING")
    cfg = load_config(config_path)
    return cfg


def _get_preprocessor(cfg):
    from src.preprocessing import MedicalImagePreprocessor
    img_size = tuple(cfg["preprocessing"]["image_size"])
    return MedicalImagePreprocessor(
        image_size=img_size,
        normalization_method=cfg["preprocessing"]["normalization_method"],
        augmentation_enabled=False,
    )


@st.cache_resource(show_spinner="Loading CNN model...")
def _load_cnn_model(cfg, _model_mtime: float = 0.0, _ae_mtime: float = 0.0):
    """Load CNN + autoencoder. Cache is keyed on file mtimes so a freshly
    trained model is automatically picked up without restarting the server."""
    from src.model import MedicalCNNModel
    from pathlib import Path

    model_path = Path(cfg["paths"]["model_save_path"])
    class_names = cfg["classes"]
    h, w = cfg["model"]["input_shape"][:2]

    cnn = MedicalCNNModel(
        input_shape=(h, w, 1),
        num_classes=len(class_names),
        class_names=class_names,
        model_save_path=model_path,
    )
    if model_path.exists():
        cnn.load_model(model_path)

    # Load autoencoder if available
    ae_path = Path(cfg["paths"]["autoencoder_save_path"])
    if ae_path.exists():
        try:
            cnn.load_autoencoder(ae_path)
        except Exception:
            pass

    return cnn


def _get_model_mtime(cfg) -> tuple:
    """Return (model_mtime, ae_mtime) — 0.0 if file does not exist yet."""
    from pathlib import Path
    model_path = Path(cfg["paths"]["model_save_path"])
    ae_path    = Path(cfg["paths"]["autoencoder_save_path"])
    return (
        model_path.stat().st_mtime if model_path.exists() else 0.0,
        ae_path.stat().st_mtime    if ae_path.exists()    else 0.0,
    )


# ---------------------------------------------------------------------------
# App Layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Medical Anomaly Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure model is trained before anything else (safe on Streamlit Cloud first launch)
_config_path_for_init = str(PROJECT_ROOT / "configs" / "config.yaml")
_ensure_model_exists(_config_path_for_init)

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0a0e1a;
        color: #e2e8f0;
    }
    .metric-card {
        background: #141c2e;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-name { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 26px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; color: white; }
    .alert-critical { background: #2d1414; border-left: 4px solid #ef4444; padding: 10px 14px; border-radius: 6px; margin: 8px 0; }
    .alert-moderate { background: #2d2314; border-left: 4px solid #f59e0b; padding: 10px 14px; border-radius: 6px; margin: 8px 0; }
    .alert-normal { background: #102414; border-left: 4px solid #10b981; padding: 10px 14px; border-radius: 6px; margin: 8px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.image("https://i.imgur.com/ZVJTEmR.png", width=40) if False else None
    st.title("🧠 MedAI Dashboard")
    st.caption("Real-time Anomaly Detection System")

    config_path = st.text_input(
        "Config path",
        value=str(PROJECT_ROOT / "configs" / "config.yaml"),
        key="cfg_path",
    )

    cfg = None
    try:
        cfg = _load_pipeline(config_path)
    except Exception as e:
        st.error(f"Config load failed: {e}")
        st.stop()

    class_names = cfg["classes"]

    # Model status indicator
    _mm, _am = _get_model_mtime(cfg)
    if _mm > 0:
        st.success("✅ Trained model detected")
    else:
        st.warning("⚠️ No trained model — run `python main.py --mode train`")

    if st.button("🔄 Reload Model", help="Force-reload model from disk after training"):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("Input Image")
    input_mode = st.radio("Source", ["Upload Image", "Synthetic Sample"], index=1)
    uploaded_file = None
    selected_scan = None

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload PNG/JPG/DCM", type=["png", "jpg", "jpeg", "bmp", "tiff"])
    else:
        selected_scan = st.selectbox(
            "Synthetic Sample",
            ["Generate Normal", "Generate Tumor", "Generate Fracture"],
        )

    st.divider()
    st.caption(f"Model: CNN | Classes: {', '.join(class_names)}")

# ---------------------------------------------------------------------------
# Load / generate image
# ---------------------------------------------------------------------------

def load_image_from_sidebar(cfg) -> Optional[np.ndarray]:
    preprocessor = _get_preprocessor(cfg)
    img_size = tuple(cfg["preprocessing"]["image_size"])

    if uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file).convert("L")
            raw = np.array(pil_img, dtype=np.float32)
            return preprocessor.preprocess_single(raw)
        except Exception as exc:
            st.error(f"Failed to load image: {exc}")
            return None

    if selected_scan is not None:
        class_map = {"Generate Normal": 0, "Generate Tumor": 1, "Generate Fracture": 2}
        class_idx = class_map[selected_scan]
        n_total = 3
        rng = np.random.default_rng(int(time.time()) % 1000)
        images, labels = preprocessor.generate_synthetic_data(
            n_samples=n_total,
            image_size=img_size,
            random_seed=int(time.time()) % 1000,
        )
        # Get first image of the desired class
        matches = images[labels == class_idx]
        if len(matches) > 0:
            return matches[0]
        return images[0]

    return None


def resize_for_model(image: np.ndarray, cfg) -> np.ndarray:
    import cv2
    h, w = cfg["model"]["input_shape"][:2]
    if image.shape[0] != h or image.shape[1] != w:
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.float32)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Real-time Detection",
    "🔭 Feature Analysis",
    "📈 Model Performance",
    "⚡ Batch Processing",
])


# ===========================================================================
# TAB 1 — Real-time Detection
# ===========================================================================

with tab1:
    st.subheader("Real-time Detection")

    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("**Input Image**")
        image_np = load_image_from_sidebar(cfg)

        if image_np is None:
            st.info("Select a sample or upload an image from the sidebar.")
        else:
            st.image(image_np, caption="Loaded image (preprocessed)", use_container_width=True, clamp=True)

            run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)

            if run_btn:
                with st.spinner("Running inference pipeline..."):
                    t0 = time.perf_counter()
                    # Pass mtimes so cache is invalidated when model is retrained
                    _mm, _am = _get_model_mtime(cfg)
                    model_img = resize_for_model(image_np, cfg)

                    try:
                        cnn = _load_cnn_model(cfg, _model_mtime=_mm, _ae_mtime=_am)
                        if cnn.model is None:
                            st.error("Model not trained yet. Run `python main.py --mode train` first.")
                        else:
                            probs = cnn.predict(model_img)
                            pred_idx = int(np.argmax(probs))
                            pred_class = class_names[pred_idx]
                            confidence = float(probs[pred_idx])
                            elapsed_ms = (time.perf_counter() - t0) * 1000

                            # Grad-CAM
                            gradcam_heatmap = None
                            try:
                                heatmap = cnn.get_gradcam(model_img, layer_name="res4_conv2")
                                from src.utils import apply_colormap_to_heatmap
                                gradcam_heatmap = apply_colormap_to_heatmap(heatmap, model_img)
                            except Exception:
                                pass

                            # Anomaly score
                            anomaly_score = 0.0
                            if cnn.autoencoder is not None:
                                try:
                                    errors = cnn.reconstruction_error(model_img[np.newaxis])
                                    anomaly_score = float(errors[0])
                                except Exception:
                                    pass

                            st.session_state["last_result"] = {
                                "probs": probs,
                                "pred_class": pred_class,
                                "pred_idx": pred_idx,
                                "confidence": confidence,
                                "anomaly_score": anomaly_score,
                                "gradcam": gradcam_heatmap,
                                "elapsed_ms": elapsed_ms,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                    except Exception as exc:
                        st.error(f"Inference error: {exc}")
                        st.code(traceback.format_exc())

            # Grad-CAM display
            if "last_result" in st.session_state and st.session_state["last_result"].get("gradcam") is not None:
                st.markdown("**Grad-CAM Heatmap Overlay**")
                st.image(
                    st.session_state["last_result"]["gradcam"],
                    caption="Activation heatmap — highlighted regions influenced prediction",
                    use_container_width=True,
                )

    with col_results:
        st.markdown("**Analysis Results**")

        if "last_result" not in st.session_state:
            st.info("Run analysis to see results here.")
        else:
            res = st.session_state["last_result"]
            pred_class = res["pred_class"]
            confidence = res["confidence"]
            anomaly_score = res["anomaly_score"]
            probs = res["probs"]

            # Risk colour
            if pred_class in ("tumor", "fracture"):
                risk_class = "alert-critical" if confidence > 0.85 else "alert-moderate"
                risk_label = "HIGH RISK" if confidence > 0.85 else "MODERATE RISK"
            else:
                risk_class = "alert-normal"
                risk_label = "LOW RISK"

            st.markdown(
                f'<div class="{risk_class}">'
                f'<b>{pred_class.upper()}</b> — {risk_label} | Confidence: {confidence:.1%}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Metric cards row
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{confidence:.1%}")
            c2.metric("Processing Time", f"{res['elapsed_ms']:.1f} ms")
            c3.metric("Anomaly Score", f"{anomaly_score:.4f}")

            st.caption(f"Timestamp: {res['timestamp']}")

            # Plotly confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={"text": f"Confidence — {pred_class.capitalize()}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00c6ff"},
                    "steps": [
                        {"range": [0, 50], "color": "#10b981"},
                        {"range": [50, 80], "color": "#f59e0b"},
                        {"range": [80, 100], "color": "#ef4444"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": 85,
                    },
                },
                number={"suffix": "%", "font": {"size": 26}},
            ))
            fig_gauge.update_layout(height=280, paper_bgcolor="#0f1526", font_color="white", margin=dict(t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Confidence bar chart for all classes
            fig_bar = px.bar(
                x=class_names,
                y=[float(p) for p in probs],
                color=class_names,
                color_discrete_sequence=["#10b981", "#ef4444", "#f59e0b"],
                labels={"x": "Class", "y": "Probability"},
                title="Class Probability Distribution",
            )
            fig_bar.update_layout(
                paper_bgcolor="#0f1526",
                plot_bgcolor="#141c2e",
                font_color="white",
                showlegend=False,
                height=260,
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Anomaly score meter
            anomaly_pct = min(anomaly_score * 1000, 100)
            fig_anom = go.Figure(go.Indicator(
                mode="gauge+number",
                value=anomaly_pct,
                title={"text": "Anomaly Score (×1000)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#f59e0b"},
                    "steps": [
                        {"range": [0, 30], "color": "#10b981"},
                        {"range": [30, 60], "color": "#f59e0b"},
                        {"range": [60, 100], "color": "#ef4444"},
                    ],
                },
                number={"suffix": "", "font": {"size": 20}},
            ))
            fig_anom.update_layout(height=220, paper_bgcolor="#0f1526", font_color="white", margin=dict(t=40, b=10))
            st.plotly_chart(fig_anom, use_container_width=True)


# ===========================================================================
# TAB 2 — Feature Analysis
# ===========================================================================

with tab2:
    st.subheader("Feature Space Analysis")

    generate_btn = st.button("Generate Feature Space (200 synthetic images)", key="feat_btn")

    if generate_btn or "feat_data" in st.session_state:
        if generate_btn:
            with st.spinner("Generating and extracting features... (this may take ~30s)"):
                try:
                    preprocessor = _get_preprocessor(cfg)
                    img_size = tuple(cfg["preprocessing"]["image_size"])
                    images, labels = preprocessor.generate_synthetic_data(
                        n_samples=60,  # smaller for speed in dashboard demo
                        image_size=img_size,
                        random_seed=42,
                    )

                    from src.feature_engineering import FeatureEngineer
                    fe = FeatureEngineer(n_pca_components=min(10, len(images) - 1))
                    feature_matrix, feature_names = fe.build_feature_matrix(list(images))
                    pca_features, _ = fe.apply_pca(feature_matrix)
                    tsne_features = fe.apply_tsne(feature_matrix, n_components=2)

                    st.session_state["feat_data"] = {
                        "feature_matrix": feature_matrix,
                        "feature_names": feature_names,
                        "pca_features": pca_features,
                        "tsne_features": tsne_features,
                        "labels": labels,
                    }
                except Exception as exc:
                    st.error(f"Feature extraction failed: {exc}")
                    st.code(traceback.format_exc())

        if "feat_data" in st.session_state:
            fd = st.session_state["feat_data"]
            labels = fd["labels"]
            label_names = [class_names[l] for l in labels]

            col_pca, col_tsne = st.columns(2)

            with col_pca:
                pca_coords = fd["pca_features"][:, :2]
                df_pca = pd.DataFrame({
                    "PC1": pca_coords[:, 0],
                    "PC2": pca_coords[:, 1],
                    "Class": label_names,
                })
                fig_pca = px.scatter(
                    df_pca, x="PC1", y="PC2", color="Class",
                    title="PCA Feature Space",
                    color_discrete_map={
                        "normal": "#10b981",
                        "tumor": "#ef4444",
                        "fracture": "#f59e0b",
                    },
                    template="plotly_dark",
                )
                fig_pca.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color="white")))
                st.plotly_chart(fig_pca, use_container_width=True)

            with col_tsne:
                tsne_coords = fd["tsne_features"]
                df_tsne = pd.DataFrame({
                    "t-SNE 1": tsne_coords[:, 0],
                    "t-SNE 2": tsne_coords[:, 1],
                    "Class": label_names,
                })
                fig_tsne = px.scatter(
                    df_tsne, x="t-SNE 1", y="t-SNE 2", color="Class",
                    title="t-SNE Feature Space",
                    color_discrete_map={
                        "normal": "#10b981",
                        "tumor": "#ef4444",
                        "fracture": "#f59e0b",
                    },
                    template="plotly_dark",
                )
                fig_tsne.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color="white")))
                st.plotly_chart(fig_tsne, use_container_width=True)

            # Top features bar chart
            st.markdown("**Top 10 Most Discriminative Features (by F-Statistic)**")
            try:
                from sklearn.feature_selection import f_classif
                fscores, _ = f_classif(fd["feature_matrix"], labels)
                fscores = np.nan_to_num(fscores)
                top_idx = np.argsort(fscores)[::-1][:10]
                df_feat = pd.DataFrame({
                    "Feature": [fd["feature_names"][i] for i in top_idx],
                    "F-Score": fscores[top_idx],
                })
                fig_feat = px.bar(
                    df_feat, x="F-Score", y="Feature",
                    orientation="h",
                    title="Top 10 Discriminative Features",
                    template="plotly_dark",
                    color="F-Score",
                    color_continuous_scale="Bluered",
                )
                fig_feat.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
                st.plotly_chart(fig_feat, use_container_width=True)
            except Exception as exc:
                st.warning(f"Feature importance failed: {exc}")

    else:
        st.info("Click the button above to generate feature space plots.")


# ===========================================================================
# TAB 3 — Model Performance
# ===========================================================================

with tab3:
    st.subheader("Model Performance Metrics")

    eval_btn = st.button("Run Evaluation on Synthetic Test Set", key="eval_btn")

    if eval_btn or "eval_data" in st.session_state:
        if eval_btn:
            with st.spinner("Evaluating model..."):
                try:
                    from src.preprocessing import MedicalImagePreprocessor
                    from src.evaluator import ModelEvaluator
                    from sklearn.model_selection import train_test_split
                    import cv2

                    preprocessor = _get_preprocessor(cfg)
                    img_size = tuple(cfg["preprocessing"]["image_size"])
                    images, labels = preprocessor.generate_synthetic_data(
                        n_samples=90,
                        image_size=img_size,
                        random_seed=99,
                    )
                    model_h, model_w = cfg["model"]["input_shape"][:2]
                    if images.shape[1] != model_h:
                        images = np.array([cv2.resize(img, (model_w, model_h)) for img in images], dtype=np.float32)

                    _, X_te, _, y_te = train_test_split(images, labels, test_size=0.35, stratify=labels, random_state=42)

                    cnn = _load_cnn_model(cfg)
                    if cnn.model is None:
                        st.error("Model not trained. Run `python main.py --mode train` first.")
                    else:
                        eval_res = cnn.evaluate(X_te, y_te)
                        evaluator = ModelEvaluator(class_names=class_names)
                        metrics = evaluator.compute_metrics(y_te, eval_res["y_pred"], eval_res["y_probs"])
                        st.session_state["eval_data"] = {
                            "metrics": metrics,
                            "y_true": y_te,
                            "y_pred": eval_res["y_pred"],
                            "y_probs": eval_res["y_probs"],
                        }
                except Exception as exc:
                    st.error(f"Evaluation error: {exc}")
                    st.code(traceback.format_exc())

        if "eval_data" in st.session_state:
            ed = st.session_state["eval_data"]
            metrics = ed["metrics"]
            y_true = ed["y_true"]
            y_pred = ed["y_pred"]
            y_probs = ed["y_probs"]

            # Metrics table
            st.markdown("**Live Metrics**")
            scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            df_metrics = pd.DataFrame(
                {"Metric": list(scalar_metrics.keys()), "Value": [f"{v:.4f}" for v in scalar_metrics.values()]}
            )
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            col_cm, col_roc = st.columns(2)

            with col_cm:
                st.markdown("**Confusion Matrix**")
                from sklearn.metrics import confusion_matrix
                import plotly.figure_factory as ff

                cm = confusion_matrix(y_true, y_pred)
                fig_cm = ff.create_annotated_heatmap(
                    z=cm,
                    x=class_names,
                    y=class_names,
                    colorscale="Blues",
                )
                fig_cm.update_layout(
                    xaxis_title="Predicted",
                    yaxis_title="True",
                    template="plotly_dark",
                    height=350,
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_roc:
                st.markdown("**ROC Curves**")
                from sklearn.metrics import roc_curve, auc
                from sklearn.preprocessing import label_binarize

                n_classes = len(class_names)
                y_bin = label_binarize(y_true, classes=list(range(n_classes)))
                colours = ["#10b981", "#ef4444", "#f59e0b"]

                fig_roc = go.Figure()
                for i, (cname, colour) in enumerate(zip(class_names, colours)):
                    try:
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                        roc_auc = auc(fpr, tpr)
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines", name=f"{cname} (AUC={roc_auc:.3f})",
                            line=dict(color=colour, width=2)
                        ))
                    except Exception:
                        pass

                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    name="Random", line=dict(dash="dash", color="gray", width=1.5)
                ))
                fig_roc.update_layout(
                    xaxis_title="FPR", yaxis_title="TPR",
                    template="plotly_dark", height=350, legend=dict(x=0.5, y=0.05),
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            # PR Curves
            st.markdown("**Precision-Recall Curves**")
            from sklearn.metrics import precision_recall_curve
            fig_pr = go.Figure()
            for i, (cname, colour) in enumerate(zip(class_names, colours)):
                try:
                    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
                    fig_pr.add_trace(go.Scatter(
                        x=recall, y=precision, mode="lines",
                        name=cname, line=dict(color=colour, width=2)
                    ))
                except Exception:
                    pass
            fig_pr.update_layout(
                xaxis_title="Recall", yaxis_title="Precision",
                template="plotly_dark", height=300,
            )
            st.plotly_chart(fig_pr, use_container_width=True)

    else:
        st.info("Click the button above to evaluate the trained model.")


# ===========================================================================
# TAB 4 — Batch Processing
# ===========================================================================

with tab4:
    st.subheader("Batch Image Processing")
    st.markdown("Upload a `.zip` archive containing PNG/JPG images to batch-process.")

    zip_file = st.file_uploader("Upload ZIP of images", type=["zip"])

    if zip_file is not None:
        process_btn = st.button("⚡ Process All Images", key="batch_btn", type="primary")

        if process_btn:
            results_list: List[Dict] = []
            progress = st.progress(0)
            status_text = st.empty()

            zf = zipfile.ZipFile(io.BytesIO(zip_file.read()))
            image_entries = [
                f for f in zf.namelist()
                if Path(f).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
            ]

            if not image_entries:
                st.warning("No supported images found in ZIP.")
            else:
                cnn = _load_cnn_model(cfg)
                preprocessor = _get_preprocessor(cfg)
                import cv2
                model_h, model_w = cfg["model"]["input_shape"][:2]

                for idx, entry in enumerate(image_entries):
                    status_text.text(f"Processing {idx + 1} / {len(image_entries)}: {Path(entry).name}")
                    try:
                        raw_bytes = zf.read(entry)
                        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("L")
                        raw = np.array(pil_img, dtype=np.float32)
                        processed = preprocessor.preprocess_single(raw)
                        if processed.shape[0] != model_h:
                            processed = cv2.resize(processed, (model_w, model_h))

                        if cnn.model is not None:
                            probs = cnn.predict(processed)
                            pred_idx = int(np.argmax(probs))
                            pred_class = class_names[pred_idx]
                            confidence = float(probs[pred_idx])
                        else:
                            probs = np.ones(len(class_names)) / len(class_names)
                            pred_class = "unknown"
                            confidence = 0.0

                        results_list.append({
                            "filename": Path(entry).name,
                            "predicted_class": pred_class,
                            "confidence": f"{confidence:.3f}",
                            **{f"prob_{cn}": f"{float(probs[i]):.3f}" for i, cn in enumerate(class_names)},
                        })
                    except Exception as exc:
                        results_list.append({
                            "filename": Path(entry).name,
                            "predicted_class": "error",
                            "confidence": "0.000",
                            "error": str(exc),
                        })
                    progress.progress((idx + 1) / len(image_entries))

                status_text.text("Processing complete!")
                df_results = pd.DataFrame(results_list)
                st.session_state["batch_results"] = df_results

        if "batch_results" in st.session_state:
            df = st.session_state["batch_results"]
            st.dataframe(df, use_container_width=True)

            # Summary stats
            if "predicted_class" in df.columns:
                st.markdown("**Summary Statistics**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Processed", len(df))
                normal_count = (df["predicted_class"] == "normal").sum()
                c2.metric("Normal", int(normal_count))
                anomaly_count = len(df) - normal_count
                c3.metric("Anomalies Detected", int(anomaly_count))

                class_dist = df["predicted_class"].value_counts()
                fig_dist = px.pie(
                    values=class_dist.values,
                    names=class_dist.index,
                    title="Class Distribution",
                    template="plotly_dark",
                    color_discrete_sequence=["#10b981", "#ef4444", "#f59e0b"],
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            # Download CSV
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Results CSV",
                data=csv_bytes,
                file_name="batch_results.csv",
                mime="text/csv",
            )
    else:
        st.info("Upload a ZIP archive to start batch processing.")
