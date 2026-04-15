"""
tests/test_preprocessing.py
----------------------------
Pytest tests for Module 1: MedicalImagePreprocessor

Tests:
  - normalize min-max output range [0, 1]
  - normalize z-score zero mean, unit std
  - resize output shape matches target
  - denoise output shape and range
  - augment output shape
  - segment_roi returns uint8 mask
  - generate_synthetic_data shape, label distribution, reproducibility
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import MedicalImagePreprocessor


@pytest.fixture()
def preprocessor() -> MedicalImagePreprocessor:
    return MedicalImagePreprocessor(image_size=(64, 64), augmentation_enabled=False, random_seed=0)


@pytest.fixture()
def sample_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((64, 64)).astype(np.float32) * 500.0  # raw scale


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_minmax_range(self, preprocessor, sample_image):
        norm = preprocessor.normalize(sample_image, method="minmax")
        assert norm.min() >= 0.0 - 1e-6, "min should be ≥ 0"
        assert norm.max() <= 1.0 + 1e-6, "max should be ≤ 1"

    def test_minmax_dtype(self, preprocessor, sample_image):
        norm = preprocessor.normalize(sample_image, method="minmax")
        assert norm.dtype == np.float32

    def test_zscore_mean_zero(self, preprocessor, sample_image):
        norm = preprocessor.normalize(sample_image, method="zscore")
        assert abs(norm.mean()) < 0.01, f"z-score mean should be ~0, got {norm.mean()}"

    def test_zscore_std_one(self, preprocessor, sample_image):
        norm = preprocessor.normalize(sample_image, method="zscore")
        assert abs(norm.std() - 1.0) < 0.05, f"z-score std should be ~1, got {norm.std()}"

    def test_uniform_image_minmax_no_crash(self, preprocessor):
        uniform = np.ones((64, 64), dtype=np.float32) * 128.0
        norm = preprocessor.normalize(uniform, method="minmax")
        assert norm.shape == (64, 64)

    def test_invalid_method_raises(self, preprocessor, sample_image):
        with pytest.raises(ValueError, match="Unknown normalization method"):
            preprocessor.normalize(sample_image, method="badmethod")


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

class TestResize:
    def test_output_shape(self, preprocessor, sample_image):
        resized = preprocessor.resize(sample_image, size=(32, 48))
        assert resized.shape == (32, 48), f"Expected (32,48), got {resized.shape}"

    def test_default_size(self, preprocessor, sample_image):
        resized = preprocessor.resize(sample_image)
        assert resized.shape == preprocessor.image_size

    def test_dtype(self, preprocessor, sample_image):
        resized = preprocessor.resize(sample_image)
        assert resized.dtype == np.float32


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

class TestDenoise:
    def test_output_shape(self, preprocessor):
        img = np.random.default_rng(1).random((64, 64)).astype(np.float32)
        denoised = preprocessor.denoise(img)
        assert denoised.shape == (64, 64)

    def test_output_range(self, preprocessor):
        img = np.random.default_rng(2).random((64, 64)).astype(np.float32)
        denoised = preprocessor.denoise(img)
        assert denoised.min() >= 0.0
        assert denoised.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class TestAugment:
    def test_output_shape_preserved(self):
        preprocessor = MedicalImagePreprocessor(image_size=(64, 64), augmentation_enabled=True, random_seed=7)
        img = np.random.default_rng(3).random((64, 64)).astype(np.float32)
        aug = preprocessor.augment(img)
        assert aug.shape == img.shape

    def test_augment_disabled_returns_copy(self):
        preprocessor = MedicalImagePreprocessor(augmentation_enabled=False)
        img = np.ones((64, 64), dtype=np.float32) * 0.5
        aug = preprocessor.augment(img)
        assert np.allclose(aug, img)


# ---------------------------------------------------------------------------
# ROI Segmentation
# ---------------------------------------------------------------------------

class TestSegmentROI:
    def test_mask_is_uint8(self, preprocessor, sample_image):
        norm = preprocessor.normalize(sample_image)
        mask = preprocessor.segment_roi(norm)
        assert mask.dtype == np.uint8

    def test_mask_shape(self, preprocessor, sample_image):
        norm = preprocessor.normalize(sample_image)
        mask = preprocessor.segment_roi(norm)
        assert mask.shape == norm.shape

    def test_mask_binary_values(self, preprocessor, sample_image):
        norm = preprocessor.normalize(sample_image)
        mask = preprocessor.segment_roi(norm)
        unique = np.unique(mask)
        assert set(unique).issubset({0, 255}), f"Mask should be binary 0/255, got {unique}"


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------

class TestGenerateSyntheticData:
    def test_shapes(self):
        preprocessor = MedicalImagePreprocessor(random_seed=42)
        images, labels = preprocessor.generate_synthetic_data(n_samples=30, image_size=(64, 64))
        assert images.shape == (30, 64, 64), f"Unexpected image shape: {images.shape}"
        assert labels.shape == (30,), f"Unexpected label shape: {labels.shape}"

    def test_label_distribution(self):
        preprocessor = MedicalImagePreprocessor(random_seed=42)
        _, labels = preprocessor.generate_synthetic_data(n_samples=30, image_size=(64, 64))
        unique, counts = np.unique(labels, return_counts=True)
        assert len(unique) == 3, f"Expected 3 classes, got {unique}"
        for c in counts:
            assert c > 0, "Each class must have at least one sample"

    def test_reproducibility(self):
        preprocessor1 = MedicalImagePreprocessor(random_seed=7)
        preprocessor2 = MedicalImagePreprocessor(random_seed=7)
        images1, labels1 = preprocessor1.generate_synthetic_data(n_samples=20, image_size=(64, 64), random_seed=7)
        images2, labels2 = preprocessor2.generate_synthetic_data(n_samples=20, image_size=(64, 64), random_seed=7)
        assert np.allclose(images1, images2), "Outputs should be identical with same seed"
        assert np.array_equal(labels1, labels2)

    def test_value_range(self):
        preprocessor = MedicalImagePreprocessor(random_seed=42)
        images, _ = preprocessor.generate_synthetic_data(n_samples=10, image_size=(64, 64))
        assert images.min() >= 0.0, "Image values must be non-negative"
        assert images.max() <= 1.0 + 1e-5, f"Image values must be ≤ 1, got max={images.max()}"

    def test_dtype(self):
        preprocessor = MedicalImagePreprocessor(random_seed=42)
        images, labels = preprocessor.generate_synthetic_data(n_samples=6, image_size=(64, 64))
        assert images.dtype == np.float32
        assert labels.dtype == np.int32
