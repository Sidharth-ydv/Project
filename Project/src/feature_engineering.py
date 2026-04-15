"""
src/feature_engineering.py
--------------------------
Module 2: Feature Engineering for Medical Image Analysis

Implements the :class:`FeatureEngineer` class, wrapping:
  - GLCM texture features (scikit-image)
  - Shape / contour features (OpenCV)
  - Per-quadrant intensity statistics
  - FFT-based frequency features
  - PCA / t-SNE / UMAP dimensionality reduction (scikit-learn / umap-learn)
  - Unified feature matrix builder
  - Feature-space visualisation (Matplotlib / saved PNG)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extract, transform, and visualise features from preprocessed medical images.

    Args:
        output_dir: Directory where visualisations are saved. Created if absent.
        n_pca_components: Number of PCA components.
        tsne_perplexity: t-SNE perplexity hyperparameter.
        tsne_n_iter: t-SNE optimisation iterations.
        umap_n_neighbors: UMAP ``n_neighbors`` hyperparameter.
        umap_min_dist: UMAP ``min_dist`` hyperparameter.
        random_state: Seed for all stochastic operations.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs",
        n_pca_components: int = 50,
        tsne_perplexity: float = 30.0,
        tsne_n_iter: int = 1000,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_pca_components = n_pca_components
        self.tsne_perplexity = tsne_perplexity
        self.tsne_n_iter = int(tsne_n_iter)
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.random_state = random_state
        self._pca: Optional[PCA] = None
        self._scaler: Optional[StandardScaler] = None
        logger.info("FeatureEngineer initialised | output_dir=%s", self.output_dir)

    # ------------------------------------------------------------------
    # Texture Features — GLCM
    # ------------------------------------------------------------------

    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Compute Gray Level Co-occurrence Matrix (GLCM) texture features.

        Properties computed: contrast, dissimilarity, homogeneity, energy,
        correlation.  Results are averaged over multiple distances and angles.

        Args:
            image: 2-D float32 array in [0, 1].

        Returns:
            Dictionary mapping feature name → scalar float value.
        """
        try:
            from skimage.feature import graycomatrix, graycoprops  # type: ignore
        except ImportError:
            from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops  # type: ignore

        # Convert to 8-level integer for GLCM efficiency
        levels = 64
        image_quantised = (image * (levels - 1)).clip(0, levels - 1).astype(np.uint8)

        distances = [1, 2]
        angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        try:
            glcm = graycomatrix(
                image_quantised,
                distances=distances,
                angles=angles,
                levels=levels,
                symmetric=True,
                normed=True,
            )
            props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
            features: Dict[str, float] = {}
            for prop in props:
                values = graycoprops(glcm, prop).flatten()
                features[f"glcm_{prop}_mean"] = float(values.mean())
                features[f"glcm_{prop}_std"] = float(values.std())
        except Exception as exc:
            logger.warning("GLCM computation failed: %s", exc)
            features = {
                f"glcm_{p}_{s}": 0.0
                for p in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
                for s in ["mean", "std"]
            }

        return features

    # ------------------------------------------------------------------
    # Shape Features — Contour-based
    # ------------------------------------------------------------------

    def extract_shape_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract geometric shape features from the largest foreground contour.

        Features: area, perimeter, circularity, aspect ratio, extent,
        solidity, equivalent diameter.

        Args:
            image: 2-D float32 array in [0, 1].

        Returns:
            Dictionary mapping feature name → scalar float.
        """
        uint8_img = (image * 255).clip(0, 255).astype(np.uint8)
        _, binary = cv2.threshold(uint8_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features: Dict[str, float] = {
            "shape_area": 0.0,
            "shape_perimeter": 0.0,
            "shape_circularity": 0.0,
            "shape_aspect_ratio": 0.0,
            "shape_extent": 0.0,
            "shape_solidity": 0.0,
            "shape_equiv_diameter": 0.0,
            "shape_num_contours": float(len(contours)),
        }

        if not contours:
            return features

        # Pick the largest contour
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        features["shape_area"] = float(area)
        features["shape_perimeter"] = float(perimeter)
        features["shape_circularity"] = (
            float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        )

        x, y, w, h = cv2.boundingRect(cnt)
        features["shape_aspect_ratio"] = float(w / h) if h > 0 else 0.0
        features["shape_extent"] = float(area / (w * h)) if (w * h) > 0 else 0.0
        features["shape_equiv_diameter"] = float(np.sqrt(4 * area / np.pi)) if area > 0 else 0.0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        features["shape_solidity"] = float(area / hull_area) if hull_area > 0 else 0.0

        return features

    # ------------------------------------------------------------------
    # Intensity Features — Per-quadrant statistics
    # ------------------------------------------------------------------

    def extract_intensity_features(self, image: np.ndarray) -> Dict[str, float]:
        """Compute statistical intensity features per quadrant.

        Statistics: mean, std, skewness, kurtosis for each of the four
        quadrants and the full image.

        Args:
            image: 2-D float32 array.

        Returns:
            Dictionary mapping feature name → scalar float.
        """
        H, W = image.shape
        half_H, half_W = H // 2, W // 2

        quadrants = {
            "q1": image[:half_H, :half_W],
            "q2": image[:half_H, half_W:],
            "q3": image[half_H:, :half_W],
            "q4": image[half_H:, half_W:],
            "full": image,
        }

        features: Dict[str, float] = {}
        for name, quad in quadrants.items():
            flat = quad.ravel().astype(np.float64)
            features[f"intensity_{name}_mean"] = float(np.mean(flat))
            features[f"intensity_{name}_std"] = float(np.std(flat))
            features[f"intensity_{name}_skew"] = float(skew(flat)) if len(flat) > 1 else 0.0
            features[f"intensity_{name}_kurtosis"] = float(kurtosis(flat)) if len(flat) > 1 else 0.0
            features[f"intensity_{name}_p25"] = float(np.percentile(flat, 25))
            features[f"intensity_{name}_p75"] = float(np.percentile(flat, 75))

        return features

    # ------------------------------------------------------------------
    # Frequency Features — FFT-based
    # ------------------------------------------------------------------

    def extract_frequency_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract dominant frequency and spectral energy features via 2-D FFT.

        Args:
            image: 2-D float32 array.

        Returns:
            Dictionary mapping feature name → scalar float.
        """
        fft = np.fft.fft2(image.astype(np.float64))
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        power = magnitude ** 2

        H, W = image.shape
        cy, cx = H // 2, W // 2

        # Radial frequency bands
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        radial_dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        low_band = (radial_dist <= min(H, W) * 0.1)
        mid_band = (radial_dist > min(H, W) * 0.1) & (radial_dist <= min(H, W) * 0.3)
        high_band = radial_dist > min(H, W) * 0.3

        total_power = power.sum() + 1e-12
        features: Dict[str, float] = {
            "freq_low_band_energy": float(power[low_band].sum() / total_power),
            "freq_mid_band_energy": float(power[mid_band].sum() / total_power),
            "freq_high_band_energy": float(power[high_band].sum() / total_power),
            "freq_total_power": float(np.log1p(total_power)),
            "freq_spectral_entropy": float(self._spectral_entropy(power / total_power)),
            "freq_dominant_frequency": float(radial_dist[np.unravel_index(power.argmax(), power.shape)]),
            "freq_mean_magnitude": float(magnitude.mean()),
            "freq_std_magnitude": float(magnitude.std()),
        }
        return features

    @staticmethod
    def _spectral_entropy(prob_map: np.ndarray) -> float:
        """Compute Shannon entropy of a normalised spectral power map."""
        flat = prob_map.ravel()
        flat = flat[flat > 1e-15]
        return float(-np.sum(flat * np.log(flat)))

    # ------------------------------------------------------------------
    # Build combined feature matrix
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self,
        image_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract all enabled features from a list of images and concatenate.

        Args:
            image_list: List of 2-D float32 NumPy arrays (already preprocessed).

        Returns:
            Tuple of:
                - ``feature_matrix`` – float32 array of shape ``(N, D)``
                - ``feature_names`` – list of ``D`` feature name strings
        """
        all_features: List[List[float]] = []
        feature_names: Optional[List[str]] = None

        for i, image in enumerate(tqdm_or_plain(image_list, desc="Extracting features")):
            try:
                feat_dict: Dict[str, float] = {}
                feat_dict.update(self.extract_texture_features(image))
                feat_dict.update(self.extract_shape_features(image))
                feat_dict.update(self.extract_intensity_features(image))
                feat_dict.update(self.extract_frequency_features(image))

                if feature_names is None:
                    feature_names = list(feat_dict.keys())

                all_features.append(list(feat_dict.values()))

            except Exception as exc:
                logger.warning("Feature extraction failed for image %d: %s", i, exc)
                # Fill with zeros if extraction fails
                if feature_names is not None:
                    all_features.append([0.0] * len(feature_names))

        if not all_features:
            raise RuntimeError("No features could be extracted from the provided images.")

        feature_matrix = np.array(all_features, dtype=np.float32)
        logger.info(
            "Feature matrix built: shape=%s, features=%d",
            feature_matrix.shape,
            len(feature_names or []),
        )
        return feature_matrix, feature_names or []

    # ------------------------------------------------------------------
    # Dimensionality Reduction
    # ------------------------------------------------------------------

    def apply_pca(
        self,
        features: np.ndarray,
        n_components: Optional[int] = None,
    ) -> Tuple[np.ndarray, PCA]:
        """Reduce feature dimensionality with PCA.

        Args:
            features: 2-D float32 array of shape ``(N, D)``.
            n_components: Number of principal components.  Defaults to
                ``self.n_pca_components``.

        Returns:
            Tuple of:
                - Transformed array of shape ``(N, n_components)``.
                - Fitted :class:`sklearn.decomposition.PCA` object.
        """
        n_components = n_components or min(self.n_pca_components, features.shape[0], features.shape[1])

        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(features)

        self._pca = PCA(n_components=n_components, random_state=self.random_state)
        transformed = self._pca.fit_transform(scaled)

        explained = self._pca.explained_variance_ratio_.cumsum()
        logger.info(
            "PCA: %d components explain %.1f%% of variance",
            n_components,
            explained[-1] * 100,
        )
        return transformed.astype(np.float32), self._pca

    def apply_tsne(
        self,
        features: np.ndarray,
        n_components: int = 2,
    ) -> np.ndarray:
        """Embed *features* into 2-D (or 3-D) space using t-SNE.

        Args:
            features: 2-D float array of shape ``(N, D)``.  Ideally already
                PCA-reduced to ≤50 dimensions for performance.
            n_components: Embedding dimensionality (typically 2).

        Returns:
            Embedded float32 array of shape ``(N, n_components)``.
        """
        perplexity = min(self.tsne_perplexity, (features.shape[0] - 1) / 3.0)
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=self.tsne_n_iter,
            random_state=self.random_state,
            learning_rate="auto",
            init="pca",
        )
        embedded = tsne.fit_transform(features)
        logger.info("t-SNE embedding complete: shape=%s", embedded.shape)
        return embedded.astype(np.float32)

    def apply_umap(
        self,
        features: np.ndarray,
        n_components: int = 2,
    ) -> np.ndarray:
        """Embed *features* into lower-dimensional space using UMAP.

        Args:
            features: 2-D float array of shape ``(N, D)``.
            n_components: Target dimensionality.

        Returns:
            Embedded float32 array of shape ``(N, n_components)``.

        Raises:
            ImportError: If ``umap-learn`` is not installed.
        """
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise ImportError("umap-learn is required: pip install umap-learn") from exc

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            random_state=self.random_state,
        )
        embedded = reducer.fit_transform(features)
        logger.info("UMAP embedding complete: shape=%s", embedded.shape)
        return embedded.astype(np.float32)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_feature_space(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[List[str]] = None,
        title_suffix: str = "",
    ) -> plt.Figure:
        """Plot PCA (2D) and t-SNE (2D) scatter plots coloured by class label.

        Args:
            features: 2-D float array of shape ``(N, D)``.
            labels: Integer label array of shape ``(N,)``.
            class_names: Display names for each integer class.
            title_suffix: Optional string appended to plot titles.

        Returns:
            :class:`matplotlib.figure.Figure` with two subplots.
        """
        class_names = class_names or [str(i) for i in range(int(labels.max()) + 1)]
        colours = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

        # PCA 2D
        n_pca = min(2, features.shape[1])
        pca2 = PCA(n_components=n_pca, random_state=self.random_state)
        scaler = StandardScaler()
        pca_coords = pca2.fit_transform(scaler.fit_transform(features))

        # t-SNE 2D
        perp = min(self.tsne_perplexity, (features.shape[0] - 1) / 3.0)
        tsne2 = TSNE(
            n_components=2,
            perplexity=perp,
            n_iter=self.tsne_n_iter,
            random_state=self.random_state,
            learning_rate="auto",
            init="pca" if features.shape[1] >= 2 else "random",
        )
        tsne_coords = tsne2.fit_transform(features)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Feature Space Visualisation {title_suffix}", fontsize=14, fontweight="bold")

        for ax, coords, method in zip(axes, [pca_coords, tsne_coords], ["PCA", "t-SNE"]):
            for ci, (cname, colour) in enumerate(zip(class_names, colours)):
                mask = labels == ci
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=[colour],
                    label=cname,
                    alpha=0.7,
                    s=30,
                    edgecolors="white",
                    linewidths=0.3,
                )
            ax.set_title(f"{method} 2-D Embedding")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        save_path = self.output_dir / "feature_space.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Feature space visualisation saved to %s", save_path)
        return fig

    def plot_top_features(
        self,
        feature_matrix: np.ndarray,
        feature_names: List[str],
        labels: np.ndarray,
        top_n: int = 10,
    ) -> plt.Figure:
        """Bar chart of the *top_n* most discriminative features by F-statistic.

        Args:
            feature_matrix: 2-D float array ``(N, D)``.
            feature_names: List of ``D`` feature names.
            labels: Integer label array ``(N,)``.
            top_n: Number of top features to display.

        Returns:
            :class:`matplotlib.figure.Figure` bar chart.
        """
        from sklearn.feature_selection import f_classif  # type: ignore

        fscores, _ = f_classif(feature_matrix, labels)
        fscores = np.nan_to_num(fscores, nan=0.0)
        top_idx = np.argsort(fscores)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(
            [feature_names[i] for i in reversed(top_idx)],
            fscores[list(reversed(top_idx))],
            color="steelblue",
            edgecolor="white",
        )
        ax.set_xlabel("F-Statistic (higher = more discriminative)")
        ax.set_title(f"Top {top_n} Most Discriminative Features")
        plt.tight_layout()
        save_path = self.output_dir / "top_features.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Top-feature chart saved to %s", save_path)
        return fig


# ---------------------------------------------------------------------------
# Internal helper (avoids hard-dep on tqdm at module level)
# ---------------------------------------------------------------------------

def tqdm_or_plain(iterable, desc: str = ""):
    """Return tqdm-wrapped iterable if available, else plain iterable."""
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(iterable, desc=desc, unit="img")
    except ImportError:
        return iterable
