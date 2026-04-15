"""
src/preprocessing.py
--------------------
Module 1: Medical Image Data Preprocessing

Provides the :class:`MedicalImagePreprocessor` class that handles loading,
normalisation, denoising, augmentation, region-of-interest segmentation,
batch processing, and synthetic data generation for training demos.

Supported formats:
  - DICOM  (.dcm)  via pydicom
  - NIfTI  (.nii / .nii.gz) via nibabel
  - Standard images (.png, .jpg, .jpeg, .bmp, .tiff) via OpenCV + PIL
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    """End-to-end preprocessing pipeline for medical imaging data.

    Args:
        image_size: Target (height, width) for resizing.  Defaults to ``(128, 128)``.
        normalization_method: ``'minmax'`` scales to [0, 1]; ``'zscore'`` uses
            per-image mean / std.  Defaults to ``'minmax'``.
        augmentation_enabled: Whether :py:meth:`augment` is non-trivially applied.
        random_seed: Seed for NumPy / Python ``random`` reproducibility.

    Attributes:
        image_size (Tuple[int, int]): Target spatial dimensions.
        normalization_method (str): Active normalisation strategy.
        augmentation_enabled (bool): Augmentation toggle.
        rng (numpy.random.Generator): Seeded NumPy RNG instance.
    """

    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
        ".dcm", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif",
        ".nii", ".gz",
    )

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        normalization_method: str = "minmax",
        augmentation_enabled: bool = True,
        random_seed: int = 42,
    ) -> None:
        self.image_size = image_size
        self.normalization_method = normalization_method.lower()
        self.augmentation_enabled = augmentation_enabled
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        random.seed(random_seed)
        logger.info(
            "MedicalImagePreprocessor initialised | size=%s | norm=%s | aug=%s",
            image_size,
            normalization_method,
            augmentation_enabled,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_dicom(self, path: Union[str, Path]) -> np.ndarray:
        """Load a medical image from disk, auto-detecting format.

        Supports DICOM (.dcm), NIfTI (.nii / .nii.gz), and standard
        raster formats (PNG, JPG, BMP, TIFF).

        Args:
            path: Absolute or relative path to the image file.

        Returns:
            2-D float32 NumPy array (grayscale, raw pixel values).

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file format is not supported.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        suffix = path.suffix.lower()

        try:
            if suffix == ".dcm":
                image = self._load_dicom_file(path)
            elif suffix in (".nii", ".gz"):
                image = self._load_nifti_file(path)
            else:
                image = self._load_standard_image(path)
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
            raise

        logger.debug("Loaded image %s | shape %s | dtype %s", path.name, image.shape, image.dtype)
        return image.astype(np.float32)

    def _load_dicom_file(self, path: Path) -> np.ndarray:
        """Internal loader for DICOM files via pydicom."""
        try:
            import pydicom  # type: ignore
        except ImportError as exc:
            raise ImportError("pydicom is required for DICOM loading: pip install pydicom") from exc

        ds = pydicom.dcmread(str(path))
        pixel_array: np.ndarray = ds.pixel_array.astype(np.float32)
        if pixel_array.ndim == 3:
            pixel_array = pixel_array[0]  # take first slice of multi-slice
        return pixel_array

    def _load_nifti_file(self, path: Path) -> np.ndarray:
        """Internal loader for NIfTI files via nibabel."""
        try:
            import nibabel as nib  # type: ignore
        except ImportError as exc:
            raise ImportError("nibabel is required for NIfTI loading: pip install nibabel") from exc

        img = nib.load(str(path))
        data: np.ndarray = np.asarray(img.dataobj).astype(np.float32)
        # Take middle axial slice if 3-D volume
        if data.ndim == 3:
            data = data[:, :, data.shape[2] // 2]
        elif data.ndim == 4:
            data = data[:, :, data.shape[2] // 2, 0]
        return data

    def _load_standard_image(self, path: Path) -> np.ndarray:
        """Internal loader for standard raster formats via OpenCV."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback to PIL for exotic formats
            pil_img = Image.open(path).convert("L")
            img = np.array(pil_img)
        return img.astype(np.float32)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize(self, image: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        """Normalise pixel intensity values.

        Args:
            image: Input 2-D float32 array.
            method: ``'minmax'`` or ``'zscore'``.  Falls back to
                ``self.normalization_method`` if ``None``.

        Returns:
            Normalised float32 array.  For *minmax* the range is [0, 1];
            for *zscore* the result is zero-mean, unit-variance.

        Raises:
            ValueError: If *method* is not ``'minmax'`` or ``'zscore'``.
        """
        method = (method or self.normalization_method).lower()

        if method == "minmax":
            lo, hi = image.min(), image.max()
            if hi - lo < 1e-8:
                logger.warning("Uniform image detected during minmax normalisation; returning zeros.")
                return np.zeros_like(image, dtype=np.float32)
            return ((image - lo) / (hi - lo)).astype(np.float32)

        elif method == "zscore":
            mean, std = image.mean(), image.std()
            if std < 1e-8:
                logger.warning("Zero-std image; returning mean-subtracted zeros.")
                return np.zeros_like(image, dtype=np.float32)
            return ((image - mean) / std).astype(np.float32)

        else:
            raise ValueError(f"Unknown normalization method: {method!r}.  Use 'minmax' or 'zscore'.")

    # ------------------------------------------------------------------
    # Resizing
    # ------------------------------------------------------------------

    def resize(
        self,
        image: np.ndarray,
        size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Resize image to target spatial dimensions.

        Args:
            image: 2-D or 3-D NumPy array.
            size: ``(height, width)`` target.  Defaults to ``self.image_size``.

        Returns:
            Resized float32 NumPy array.
        """
        target_h, target_w = size or self.image_size
        resized = cv2.resize(
            image.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized.astype(np.float32)

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------

    def denoise(
        self,
        image: np.ndarray,
        gaussian_kernel: Tuple[int, int] = (5, 5),
        gaussian_sigma: float = 1.0,
        median_kernel: int = 5,
    ) -> np.ndarray:
        """Apply Gaussian followed by Median denoising.

        Args:
            image: 2-D float32 array in [0, 1].
            gaussian_kernel: Kernel size for Gaussian blur.
            gaussian_sigma: Standard deviation for Gaussian kernel.
            median_kernel: Aperture size for median filter (must be odd).

        Returns:
            Denoised float32 array with same shape as *image*.
        """
        # Gaussian
        blurred = cv2.GaussianBlur(image, gaussian_kernel, gaussian_sigma)
        # Median requires uint8 input
        uint8_img = (blurred * 255).clip(0, 255).astype(np.uint8)
        median_filtered = cv2.medianBlur(uint8_img, median_kernel)
        return (median_filtered / 255.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def augment(
        self,
        image: np.ndarray,
        rotation_range: float = 20.0,
        brightness_range: Tuple[float, float] = (0.75, 1.25),
    ) -> np.ndarray:
        """Apply a rich set of stochastic augmentations for training data variety.

        Operations applied:
            - Horizontal flip (50%)
            - Vertical flip (30%)
            - Random rotation ±*rotation_range* degrees
            - Brightness / contrast jitter
            - Gaussian noise injection
            - Random zoom / crop
            - Elastic grid distortion (30%)

        Args:
            image: 2-D float32 image in [0, 1].
            rotation_range: Maximum absolute rotation angle in degrees.
            brightness_range: (min, max) multiplicative brightness factor.

        Returns:
            Augmented float32 array, same shape as *image*.
        """
        if not self.augmentation_enabled:
            return image

        aug = image.copy()
        h, w = aug.shape[:2]

        # Horizontal flip
        if self.rng.random() > 0.5:
            aug = np.fliplr(aug)

        # Vertical flip
        if self.rng.random() > 0.7:
            aug = np.flipud(aug)

        # Random rotation
        angle = float(self.rng.uniform(-rotation_range, rotation_range))
        M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Random zoom (crop + resize back)
        if self.rng.random() > 0.5:
            zoom = float(self.rng.uniform(0.85, 1.0))
            crop_h, crop_w = int(h * zoom), int(w * zoom)
            top  = int(self.rng.integers(0, h - crop_h + 1))
            left = int(self.rng.integers(0, w - crop_w + 1))
            aug = aug[top:top + crop_h, left:left + crop_w]
            aug = cv2.resize(aug, (w, h), interpolation=cv2.INTER_LINEAR)

        # Brightness & contrast jitter
        alpha = float(self.rng.uniform(brightness_range[0], brightness_range[1]))  # contrast
        beta  = float(self.rng.uniform(-0.10, 0.10))                               # brightness
        aug = (aug * alpha + beta).clip(0.0, 1.0).astype(np.float32)

        # Gaussian noise
        if self.rng.random() > 0.5:
            noise_std = float(self.rng.uniform(0.005, 0.025))
            aug = (aug + self.rng.normal(0, noise_std, aug.shape)).clip(0.0, 1.0).astype(np.float32)

        # Elastic distortion (light)
        if self.rng.random() > 0.7:
            aug = self._elastic_distort(aug, alpha=8.0, sigma=4.0)

        return aug

    def _elastic_distort(self, image: np.ndarray, alpha: float = 8.0, sigma: float = 4.0) -> np.ndarray:
        """Apply elastic grid distortion to an image."""
        h, w = image.shape[:2]
        dx = cv2.GaussianBlur(
            self.rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma
        ) * alpha
        dy = cv2.GaussianBlur(
            self.rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma
        ) * alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # ------------------------------------------------------------------
    # ROI Segmentation
    # ------------------------------------------------------------------

    def segment_roi(self, image: np.ndarray, threshold_factor: float = 0.3) -> np.ndarray:
        """Extract a binary region-of-interest mask via Otsu thresholding.

        Args:
            image: 2-D float32 image in [0, 1].
            threshold_factor: If Otsu fails, use this fraction of max intensity.

        Returns:
            Binary ``uint8`` mask (255 = ROI, 0 = background), same spatial
            dimensions as *image*.
        """
        uint8_img = (image * 255).clip(0, 255).astype(np.uint8)
        _, mask = cv2.threshold(uint8_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if mask.sum() == 0:
            # Otsu gave empty mask — fallback
            fallback_thresh = int(threshold_factor * 255)
            _, mask = cv2.threshold(uint8_img, fallback_thresh, 255, cv2.THRESH_BINARY)
            logger.debug("Otsu threshold produced empty mask; using fallback threshold %d", fallback_thresh)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    # ------------------------------------------------------------------
    # Full preprocessing pipeline (single image)
    # ------------------------------------------------------------------

    def preprocess_single(self, image: np.ndarray) -> np.ndarray:
        """Apply the full preprocessing chain to a single image array.

        Steps: resize → denoise → normalize.

        Args:
            image: Raw 2-D float32 NumPy array.

        Returns:
            Preprocessed float32 array of shape ``(*self.image_size,)``.
        """
        img = self.resize(image)
        img = self.denoise(img)
        img = self.normalize(img)
        return img

    # ------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------

    def preprocess_batch(
        self,
        folder_path: Union[str, Path],
        output_folder: Optional[Union[str, Path]] = None,
        save_npy: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """Preprocess all supported images found in *folder_path*.

        Args:
            folder_path: Directory containing raw medical images.
            output_folder: Optional directory to save ``.npy`` files.
            save_npy: Whether to persist each processed image as ``.npy``.

        Returns:
            Tuple of:
                - ``images`` – float32 array of shape
                  ``(N, H, W)`` where ``H, W = self.image_size``.
                - ``filenames`` – list of source filenames.
        """
        folder_path = Path(folder_path).resolve()
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        files = sorted(
            p for p in folder_path.iterdir()
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )
        logger.info("Found %d image files in %s", len(files), folder_path)

        images: List[np.ndarray] = []
        filenames: List[str] = []

        for fp in tqdm(files, desc="Preprocessing images", unit="img"):
            try:
                raw = self.load_dicom(fp)
                processed = self.preprocess_single(raw)
                images.append(processed)
                filenames.append(fp.name)

                if save_npy and output_folder is not None:
                    out_dir = Path(output_folder).resolve()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    self.save_processed(processed, out_dir / (fp.stem + ".npy"))

            except Exception as exc:
                logger.warning("Skipping %s due to error: %s", fp.name, exc)
                continue

        logger.info("Successfully preprocessed %d / %d images", len(images), len(files))
        return np.array(images, dtype=np.float32), filenames

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_processed(self, image: np.ndarray, output_path: Union[str, Path]) -> None:
        """Save a preprocessed image array to disk as a ``.npy`` file.

        Args:
            image: NumPy array to serialise.
            output_path: Target file path (extension replaced or appended with .npy).

        Returns:
            None
        """
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), image)
        logger.debug("Saved preprocessed image → %s", output_path)

    # ------------------------------------------------------------------
    # Synthetic Data Generator
    # ------------------------------------------------------------------

    def generate_synthetic_data(
        self,
        n_samples: int = 200,
        image_size: Tuple[int, int] = (128, 128),
        output_dir: Optional[Union[str, Path]] = None,
        random_seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate reproducible synthetic grayscale MRI-like images with labels.

        Three-class generation strategy:

        - **normal** (label 0): Gaussian blob background + mild random noise
          simulating normal brain tissue texture.
        - **tumor** (label 1): Normal background plus 1–3 bright elliptical blobs
          at random positions, mimicking hyperintense lesions.
        - **fracture** (label 2): Normal background plus 1–2 dark rectangular/line
          patches, mimicking cortical disruptions.

        Args:
            n_samples: Total number of synthetic samples (equally split across classes).
            image_size: ``(height, width)`` of generated images.
            output_dir: If provided, each image is saved as a PNG in class-named
                sub-directories under *output_dir*.
            random_seed: Override instance seed for reproducibility.

        Returns:
            Tuple of:
                - ``images`` – float32 array of shape ``(n_samples, H, W)``,
                  values in [0, 1].
                - ``labels`` – int32 array of shape ``(n_samples,)`` with
                  class indices {0, 1, 2}.
        """
        seed = random_seed if random_seed is not None else self.random_seed
        rng = np.random.default_rng(seed)

        class_names = ["normal", "tumor", "fracture"]
        n_per_class = n_samples // len(class_names)
        remainder = n_samples - n_per_class * len(class_names)
        counts = [n_per_class] * len(class_names)
        counts[0] += remainder  # add any remainder to 'normal'

        H, W = image_size
        all_images: List[np.ndarray] = []
        all_labels: List[int] = []

        if output_dir is not None:
            out_base = Path(output_dir).resolve()
            for cn in class_names:
                (out_base / cn).mkdir(parents=True, exist_ok=True)

        for class_idx, (class_name, n_class) in enumerate(zip(class_names, counts)):
            logger.info("Generating %d synthetic '%s' images...", n_class, class_name)

            for i in range(n_class):
                img = self._generate_base_tissue(rng, H, W)

                if class_name == "tumor":
                    img = self._add_tumor_blobs(rng, img)
                elif class_name == "fracture":
                    img = self._add_fracture_patches(rng, img)

                # Final range clamp and normalise
                img = img.clip(0.0, 1.0).astype(np.float32)

                all_images.append(img)
                all_labels.append(class_idx)

                if output_dir is not None:
                    self._save_synthetic_png(img, out_base / class_name / f"{class_name}_{i:04d}.png")

        images = np.array(all_images, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int32)

        # Shuffle deterministically
        shuffle_idx = rng.permutation(len(images))
        images = images[shuffle_idx]
        labels = labels[shuffle_idx]

        logger.info(
            "Synthetic dataset created: %d images | classes: %s",
            len(images),
            {cn: int((labels == ci).sum()) for ci, cn in enumerate(class_names)},
        )
        return images, labels

    # ---------- Internal synthetic helpers ----------

    def _generate_base_tissue(self, rng: np.random.Generator, H: int, W: int) -> np.ndarray:
        """Generate a high-quality MRI-like brain tissue background."""
        cy, cx = H / 2.0, W / 2.0
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_dist = np.sqrt(cy ** 2 + cx ** 2)

        # Elliptical brain shape (not just circular)
        ry_skull = cy * rng.uniform(0.82, 0.94)
        rx_skull = cx * rng.uniform(0.82, 0.94)
        skull_mask = ((yy - cy) / ry_skull) ** 2 + ((xx - cx) / rx_skull) ** 2 <= 1.0

        # Radial gradient (brighter centre, darker periphery)
        gradient = 1.0 - 0.45 * (dist / max_dist)

        # Add subtle gyrus-like texture with 2 scales of noise
        coarse_noise = rng.normal(0, 0.04, (H, W)).astype(np.float32)
        fine_noise   = rng.normal(0, 0.02, (H, W)).astype(np.float32)
        # Smooth coarse noise → sulcal pattern
        coarse_noise = cv2.GaussianBlur(coarse_noise, (15, 15), 3)
        img = (gradient + coarse_noise + fine_noise).clip(0.05, 1.0).astype(np.float32)

        # Inner grey-matter ring (slightly brighter band)
        ry_gm = ry_skull * rng.uniform(0.55, 0.70)
        rx_gm = rx_skull * rng.uniform(0.55, 0.70)
        gm_mask = ((yy - cy) / ry_gm) ** 2 + ((xx - cx) / rx_gm) ** 2 <= 1.0
        img[skull_mask & ~gm_mask] = (img[skull_mask & ~gm_mask] * rng.uniform(1.05, 1.15)).clip(0, 1)

        # Ventricle-like dark region near centre
        vx_off = rng.uniform(-0.08, 0.08) * W
        vy_off = rng.uniform(-0.05, 0.05) * H
        rv_y = cy * rng.uniform(0.08, 0.14)
        rv_x = cx * rng.uniform(0.14, 0.22)
        vent_mask = ((yy - (cy + vy_off)) / rv_y) ** 2 + ((xx - (cx + vx_off)) / rv_x) ** 2 <= 1.0
        img[vent_mask] = rng.uniform(0.05, 0.18)

        # Background (outside skull) → near-black
        img[~skull_mask] = rng.uniform(0.0, 0.05, img[~skull_mask].shape).astype(np.float32)
        return img.clip(0.0, 1.0).astype(np.float32)

    def _add_tumor_blobs(self, rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
        """Add 1–3 hyper-intense lesion blobs with ring enhancement + mass-effect."""
        H, W = img.shape
        n_blobs = int(rng.integers(1, 4))
        result = img.copy()

        for _ in range(n_blobs):
            # Keep tumour inside brain (central 60%)
            cy = int(rng.integers(int(H * 0.2), int(H * 0.8)))
            cx = int(rng.integers(int(W * 0.2), int(W * 0.8)))
            ry = int(rng.integers(max(6, H // 10), H // 5))
            rx = int(rng.integers(max(6, W // 10), W // 5))
            core_intensity   = float(rng.uniform(0.85, 1.00))   # bright core
            ring_intensity   = float(rng.uniform(0.60, 0.80))   # ring enhancement
            oedema_intensity = float(rng.uniform(0.60, 0.75))   # surrounding oedema

            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            norm_dist = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)

            # Oedema halo (2× radius)
            oedema_mask = norm_dist <= 2.0
            result[oedema_mask] = np.maximum(
                result[oedema_mask],
                oedema_intensity * (1 - 0.25 * norm_dist[oedema_mask].clip(0, 2)),
            )

            # Ring enhancement (0.65–1.0 radius)
            ring_mask = (norm_dist >= 0.65) & (norm_dist <= 1.0)
            result[ring_mask] = np.maximum(result[ring_mask], ring_intensity)

            # Core (necrotic — dark centre 0–0.4 radius)
            core_mask = norm_dist <= 0.40
            result[core_mask] = rng.uniform(0.08, 0.22, result[core_mask].shape).astype(np.float32)

            # Bright core wall (0.40–0.65 radius)
            wall_mask = (norm_dist > 0.40) & (norm_dist < 0.65)
            result[wall_mask] = np.maximum(result[wall_mask], core_intensity)

        return result.clip(0.0, 1.0).astype(np.float32)

    def _add_fracture_patches(self, rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
        """Add 1–3 dark fracture lines with bright surrounding haemorrhage / callus."""
        H, W = img.shape
        n_patches = int(rng.integers(1, 4))
        result = img.copy()

        for _ in range(n_patches):
            # Fracture line parameters
            y1 = int(rng.integers(H // 6, 5 * H // 6))
            x1 = int(rng.integers(W // 6, 5 * W // 6))
            line_len  = int(rng.integers(W // 6, W // 2))
            angle_rad = float(rng.uniform(0, np.pi))
            thickness = int(rng.integers(2, 5))
            line_dark = float(rng.uniform(0.02, 0.10))  # very dark fracture line

            x2 = int(x1 + line_len * np.cos(angle_rad))
            y2 = int(y1 + line_len * np.sin(angle_rad))

            # Draw dark fracture
            temp = (result * 255).clip(0, 255).astype(np.uint8)
            cv2.line(temp, (x1, y1), (x2, y2), int(line_dark * 255), thickness)

            # Bright haemorrhage halo (dilate the line)
            fracture_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.line(fracture_mask, (x1, y1), (x2, y2), 255, thickness)
            halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            halo_mask = cv2.dilate(fracture_mask, halo_kernel) - fracture_mask
            halo_val = int(rng.uniform(0.72, 0.90) * 255)
            temp[halo_mask > 0] = np.maximum(temp[halo_mask > 0], halo_val)

            result = temp.astype(np.float32) / 255.0

        return result.clip(0.0, 1.0).astype(np.float32)

    def _save_synthetic_png(self, image: np.ndarray, path: Path) -> None:
        """Save a float [0,1] image as an 8-bit PNG."""
        uint8_img = (image * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(path), uint8_img)
