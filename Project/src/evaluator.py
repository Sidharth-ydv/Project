"""
src/evaluator.py
----------------
Module 5: Model Evaluation and Reporting

Implements :class:`ModelEvaluator` with:
  - Multi-class metrics (accuracy, precision, recall, F1, ROC-AUC, Kappa, MCC)
  - Publication-quality plots (confusion matrix, ROC, PR curves, training history)
  - Stratified K-fold cross-validation with std deviation
  - PDF evaluation report generation
  - Latency / throughput benchmarking
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

# Seaborn theme
sns.set_theme(style="whitegrid", palette="tab10")


class ModelEvaluator:
    """Comprehensive evaluation toolkit for the medical imaging anomaly detection model.

    Args:
        class_names: Display names for each class label (ordered by integer index).
        output_dir: Root directory to save all generated figures and reports.
        dpi: DPI for saved plot images.
        random_state: Seed for cross-validation splitting.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        output_dir: Union[str, Path] = "outputs",
        dpi: int = 150,
        random_state: int = 42,
    ) -> None:
        self.class_names = class_names or ["normal", "tumor", "fracture"]
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.random_state = random_state
        logger.info("ModelEvaluator ready | classes=%s | output=%s", self.class_names, self.output_dir)

    # ------------------------------------------------------------------
    # Scalar Metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute a comprehensive set of classification metrics.

        Args:
            y_true: Ground-truth integer label array ``(N,)``.
            y_pred: Predicted integer label array ``(N,)``.
            y_probs: Predicted probability matrix ``(N, C)`` for AUC computation.
                If ``None``, AUC is omitted.

        Returns:
            Dictionary of metric names → float values:
                ``accuracy``, ``precision``, ``recall``, ``f1``,
                ``cohen_kappa``, ``mcc``, and optionally ``roc_auc``.
        """
        y_true = np.asarray(y_true, dtype=np.int32)
        y_pred = np.asarray(y_pred, dtype=np.int32)

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }

        if y_probs is not None:
            try:
                n_classes = len(self.class_names)
                if n_classes == 2:
                    auc_score = roc_auc_score(y_true, y_probs[:, 1])
                else:
                    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
                    auc_score = roc_auc_score(y_bin, y_probs, multi_class="ovr", average="weighted")
                metrics["roc_auc"] = float(auc_score)
            except Exception as exc:
                logger.warning("AUC computation failed: %s", exc)

        logger.info(
            "Metrics | acc=%.3f | f1=%.3f | kappa=%.3f | mcc=%.3f",
            metrics["accuracy"],
            metrics["f1"],
            metrics["cohen_kappa"],
            metrics["mcc"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalise: bool = True,
        filename: str = "confusion_matrix.png",
    ) -> plt.Figure:
        """Plot and save a Seaborn confusion matrix heatmap.

        Args:
            y_true: Ground-truth integer labels.
            y_pred: Predicted integer labels.
            class_names: Override class display names.
            normalise: If ``True``, show row-normalised percentages.
            filename: Output PNG filename within ``self.output_dir``.

        Returns:
            :class:`matplotlib.figure.Figure` instance.
        """
        cnames = class_names or self.class_names
        cm = confusion_matrix(y_true, y_pred)

        if normalise:
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
            vmin, vmax = 0.0, 1.0
        else:
            cm_display = cm.astype(float)
            fmt = ".0f"
            vmin, vmax = None, None

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=cnames,
            yticklabels=cnames,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_ylabel("True Label", fontweight="bold")
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_title("Confusion Matrix" + (" (Normalised)" if normalise else ""), fontweight="bold")
        plt.tight_layout()

        save_path = self.output_dir / filename
        fig.savefig(str(save_path), dpi=self.dpi, bbox_inches="tight")
        logger.info("Confusion matrix saved → %s", save_path)
        return fig

    # ------------------------------------------------------------------
    # ROC Curves
    # ------------------------------------------------------------------

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        class_names: Optional[List[str]] = None,
        filename: str = "roc_curves.png",
    ) -> plt.Figure:
        """Plot one-vs-rest ROC curves for each class.

        Args:
            y_true: Ground-truth integer label array ``(N,)``.
            y_probs: Predicted probabilities ``(N, C)``.
            class_names: Override class display names.
            filename: Output PNG filename.

        Returns:
            :class:`matplotlib.figure.Figure` with one curve per class.
        """
        cnames = class_names or self.class_names
        n_classes = len(cnames)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(8, 6))
        colours = plt.cm.tab10(np.linspace(0, 1, n_classes))

        for i, (cname, colour) in enumerate(zip(cnames, colours)):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, color=colour, label=f"{cname} (AUC={roc_auc:.3f})")
            except Exception as exc:
                logger.warning("ROC curve failed for class %s: %s", cname, exc)

        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.set_title("ROC Curves (One-vs-Rest)", fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / filename
        fig.savefig(str(save_path), dpi=self.dpi, bbox_inches="tight")
        logger.info("ROC curves saved → %s", save_path)
        return fig

    # ------------------------------------------------------------------
    # Precision-Recall Curves
    # ------------------------------------------------------------------

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        class_names: Optional[List[str]] = None,
        filename: str = "precision_recall_curves.png",
    ) -> plt.Figure:
        """Plot per-class precision-recall curves with average precision scores.

        Args:
            y_true: Ground-truth integer labels.
            y_probs: Predicted probabilities ``(N, C)``.
            class_names: Override class display names.
            filename: Output PNG filename.

        Returns:
            :class:`matplotlib.figure.Figure` instance.
        """
        cnames = class_names or self.class_names
        n_classes = len(cnames)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(8, 6))
        colours = plt.cm.tab10(np.linspace(0, 1, n_classes))

        for i, (cname, colour) in enumerate(zip(cnames, colours)):
            try:
                precision, recall, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
                ap = float(np.trapz(precision[::-1], recall[::-1]))
                ax.plot(recall, precision, lw=2, color=colour, label=f"{cname} (AP={ap:.3f})")
            except Exception as exc:
                logger.warning("PR curve failed for class %s: %s", cname, exc)

        ax.set_xlabel("Recall", fontweight="bold")
        ax.set_ylabel("Precision", fontweight="bold")
        ax.set_title("Precision-Recall Curves", fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / filename
        fig.savefig(str(save_path), dpi=self.dpi, bbox_inches="tight")
        logger.info("PR curves saved → %s", save_path)
        return fig

    # ------------------------------------------------------------------
    # Training History
    # ------------------------------------------------------------------

    def plot_training_history(
        self,
        history: Any,
        filename: str = "training_history.png",
    ) -> plt.Figure:
        """Plot loss and accuracy over training epochs.

        Args:
            history: Keras ``History`` object (has ``.history`` dict attribute)
                or plain dict with keys ``loss``, ``accuracy``, optionally
                ``val_loss``, ``val_accuracy``.
            filename: Output PNG filename.

        Returns:
            :class:`matplotlib.figure.Figure` with two subplots.
        """
        h = history.history if hasattr(history, "history") else history

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Training History", fontweight="bold", fontsize=13)

        # Loss
        ax = axes[0]
        ax.plot(h["loss"], label="Train Loss", lw=2)
        if "val_loss" in h:
            ax.plot(h["val_loss"], label="Val Loss", lw=2, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Cross-Entropy Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy
        ax = axes[1]
        ax.plot(h.get("accuracy", h.get("acc", [])), label="Train Accuracy", lw=2)
        val_key = "val_accuracy" if "val_accuracy" in h else "val_acc"
        if val_key in h:
            ax.plot(h[val_key], label="Val Accuracy", lw=2, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / filename
        fig.savefig(str(save_path), dpi=self.dpi, bbox_inches="tight")
        logger.info("Training history saved → %s", save_path)
        return fig

    # ------------------------------------------------------------------
    # Cross-Validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 5,
        epochs: int = 20,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Stratified K-fold cross-validation with mean and std reporting.

        Args:
            model: :class:`~src.model.MedicalCNNModel` instance (will be rebuilt
                and trained fresh for each fold).
            X: Feature/image array ``(N, H, W)`` or ``(N, H, W, C)``.
            y: Integer label array ``(N,)``.
            k: Number of stratified folds.
            epochs: Training epochs per fold.
            batch_size: Mini-batch size per fold.

        Returns:
            Dictionary with ``mean_accuracy``, ``std_accuracy``, ``mean_f1``,
            ``std_f1``, ``fold_results``.
        """
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        fold_metrics: List[Dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info("CV Fold %d / %d ...", fold_idx + 1, k)
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Rebuild model for clean state
            model.build_model()
            model.train(
                X_tr, y_tr,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.0,
                patience=5,
            )

            _, y_probs = model.predict_batch(X_val)
            y_pred_fold = np.argmax(y_probs, axis=1)
            fm = self.compute_metrics(y_val, y_pred_fold, y_probs)
            fold_metrics.append(fm)
            logger.info("Fold %d | acc=%.3f | f1=%.3f", fold_idx + 1, fm["accuracy"], fm["f1"])

        accs = np.array([m["accuracy"] for m in fold_metrics])
        f1s = np.array([m["f1"] for m in fold_metrics])

        result = {
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std()),
            "mean_f1": float(f1s.mean()),
            "std_f1": float(f1s.std()),
            "fold_results": fold_metrics,  # type: ignore
        }
        logger.info(
            "Cross-validation complete | acc=%.3f±%.3f | f1=%.3f±%.3f",
            result["mean_accuracy"],
            result["std_accuracy"],
            result["mean_f1"],
            result["std_f1"],
        )
        return result

    # ------------------------------------------------------------------
    # PDF Report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        all_metrics: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        figures: Optional[Dict[str, plt.Figure]] = None,
    ) -> Path:
        """Generate a PDF evaluation report using FPDF2.

        Args:
            all_metrics: Dictionary of metric name → value (strings or floats).
            output_path: Destination PDF path. Defaults to
                ``self.output_dir / 'evaluation_report.pdf'``.
            figures: Optional dict mapping title → matplotlib Figure to embed.

        Returns:
            Resolved :class:`pathlib.Path` of the saved PDF.

        Raises:
            ImportError: If ``fpdf2`` is not installed.
        """
        try:
            from fpdf import FPDF  # type: ignore
        except ImportError as exc:
            raise ImportError("fpdf2 is required for PDF reports: pip install fpdf2") from exc

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 14, "Medical Anomaly Detection — Evaluation Report", ln=True, align="C")
        pdf.ln(4)

        import datetime
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(6)

        # Metrics table
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Classification Metrics", ln=True)

        pdf.set_font("Helvetica", size=11)
        row_h = 8
        for key, val in all_metrics.items():
            if isinstance(val, (int, float)):
                pdf.cell(80, row_h, str(key), border=1)
                pdf.cell(60, row_h, f"{val:.4f}", border=1, ln=True)
            else:
                continue  # skip nested dicts for simple table

        pdf.ln(6)

        # Embed figures as PNG
        if figures:
            for title, fig in figures.items():
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 13)
                pdf.cell(0, 10, title, ln=True)

                buf = io.BytesIO()
                fig.savefig(buf, format="PNG", dpi=120, bbox_inches="tight")
                buf.seek(0)

                tmp_png = self.output_dir / f"_tmp_{title.replace(' ', '_')}.png"
                tmp_png.write_bytes(buf.read())
                pdf.image(str(tmp_png), x=10, w=180)
                tmp_png.unlink(missing_ok=True)

        out = Path(output_path or (self.output_dir / "evaluation_report.pdf")).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        pdf.output(str(out))
        logger.info("PDF report saved → %s", out)
        return out

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark_speed(
        self,
        processor: Any,
        n_images: int = 100,
        image_size: Tuple[int, int] = (128, 128),
    ) -> Dict[str, float]:
        """Measure throughput and latency percentiles over synthetic images.

        Args:
            processor: :class:`~src.realtime_processor.RealTimeProcessor` instance.
            n_images: Number of synthetic test images to process.
            image_size: Spatial dimensions for the synthetic images.

        Returns:
            Dictionary with:
                ``throughput_per_sec``, ``mean_latency_ms``,
                ``p50_ms``, ``p95_ms``, ``p99_ms``, ``total_time_s``.
        """
        rng = np.random.default_rng(seed=0)
        images = [rng.random(image_size, dtype=np.float64).astype(np.float32) for _ in range(n_images)]

        latencies: List[float] = []
        t_total_start = time.perf_counter()

        for img in images:
            t0 = time.perf_counter()
            try:
                processor.processing_pipeline(img)
            except Exception as exc:
                logger.debug("Benchmark image failed: %s", exc)
            latencies.append((time.perf_counter() - t0) * 1_000)

        total_s = time.perf_counter() - t_total_start
        arr = np.array(latencies, dtype=np.float32)

        result = {
            "throughput_per_sec": n_images / total_s,
            "mean_latency_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "total_time_s": total_s,
        }

        logger.info(
            "Benchmark: %.1f img/s | p50=%.1f ms | p95=%.1f ms | p99=%.1f ms",
            result["throughput_per_sec"],
            result["p50_ms"],
            result["p95_ms"],
            result["p99_ms"],
        )
        return result
