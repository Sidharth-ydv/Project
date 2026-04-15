"""
tests/test_model.py
-------------------
Pytest tests for Module 3: MedicalCNNModel

Tests:
  - build_model constructs without error
  - model output shape matches num_classes
  - predict returns probabilities summing to 1.0
  - build_autoencoder constructs without error
  - autoencoder reconstruction output shape matches input
  - build_hybrid_model constructs without error
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="module")
def cnn_model():
    """Build and return a MedicalCNNModel instance (CNN only — no training)."""
    from src.model import MedicalCNNModel
    model = MedicalCNNModel(
        input_shape=(64, 64, 1),
        num_classes=3,
        class_names=["normal", "tumor", "fracture"],
        model_save_path="models/test_model.keras",
    )
    model.build_model()
    model.compile_model(learning_rate=1e-3)
    return model


@pytest.fixture(scope="module")
def sample_image_64() -> np.ndarray:
    rng = np.random.default_rng(99)
    return rng.random((64, 64)).astype(np.float32)


@pytest.fixture(scope="module")
def sample_batch_64() -> np.ndarray:
    rng = np.random.default_rng(100)
    return rng.random((8, 64, 64)).astype(np.float32)


# ---------------------------------------------------------------------------
# Build Tests
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_builds_without_error(self, cnn_model):
        assert cnn_model.model is not None

    def test_output_layer_units(self, cnn_model):
        output_shape = cnn_model.model.output_shape
        assert output_shape[-1] == 3, f"Expected 3 output units, got {output_shape[-1]}"

    def test_model_has_expected_layers(self, cnn_model):
        layer_names = [l.name for l in cnn_model.model.layers]
        assert "conv1" in layer_names
        assert "conv4" in layer_names
        assert "predictions" in layer_names

    def test_param_count_nonzero(self, cnn_model):
        assert cnn_model.model.count_params() > 0


class TestBuildAutoencoder:
    def test_autoencoder_builds(self):
        from src.model import MedicalCNNModel
        m = MedicalCNNModel(input_shape=(64, 64, 1), num_classes=3)
        ae, enc, dec = m.build_autoencoder(input_shape=(64, 64, 1))
        assert ae is not None
        assert enc is not None

    def test_autoencoder_output_shape(self):
        from src.model import MedicalCNNModel
        m = MedicalCNNModel(input_shape=(64, 64, 1), num_classes=3)
        ae, _, _ = m.build_autoencoder(input_shape=(64, 64, 1))
        expected = (None, 64, 64, 1)
        assert ae.output_shape == expected, f"Expected {expected}, got {ae.output_shape}"


class TestBuildHybridModel:
    def test_hybrid_builds(self):
        from src.model import MedicalCNNModel
        m = MedicalCNNModel(input_shape=(64, 64, 1), num_classes=3)
        hybrid = m.build_hybrid_model(n_engineered_features=20, input_shape=(64, 64, 1))
        assert hybrid is not None
        assert len(hybrid.inputs) == 2  # image + features


# ---------------------------------------------------------------------------
# Predict Tests
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_array(self, cnn_model, sample_image_64):
        probs = cnn_model.predict(sample_image_64)
        assert isinstance(probs, np.ndarray)

    def test_predict_shape(self, cnn_model, sample_image_64):
        probs = cnn_model.predict(sample_image_64)
        assert probs.shape == (3,), f"Expected (3,), got {probs.shape}"

    def test_predict_sums_to_one(self, cnn_model, sample_image_64):
        probs = cnn_model.predict(sample_image_64)
        assert abs(probs.sum() - 1.0) < 1e-5, f"Probs sum to {probs.sum()}, expected ~1.0"

    def test_predict_non_negative(self, cnn_model, sample_image_64):
        probs = cnn_model.predict(sample_image_64)
        assert np.all(probs >= 0.0)

    def test_predict_at_most_one(self, cnn_model, sample_image_64):
        probs = cnn_model.predict(sample_image_64)
        assert np.all(probs <= 1.0 + 1e-6)


class TestPredictBatch:
    def test_batch_predict_shapes(self, cnn_model, sample_batch_64):
        indices, probs = cnn_model.predict_batch(sample_batch_64)
        assert indices.shape == (8,), f"Expected (8,), got {indices.shape}"
        assert probs.shape == (8, 3), f"Expected (8,3), got {probs.shape}"

    def test_batch_probs_sum_to_one(self, cnn_model, sample_batch_64):
        _, probs = cnn_model.predict_batch(sample_batch_64)
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), f"Row sums: {row_sums}"

    def test_batch_indices_valid(self, cnn_model, sample_batch_64):
        indices, _ = cnn_model.predict_batch(sample_batch_64)
        assert np.all(indices >= 0) and np.all(indices < 3)


# ---------------------------------------------------------------------------
# Compile / Training (micro-train sanity)
# ---------------------------------------------------------------------------

class TestCompile:
    def test_compile_does_not_raise(self):
        from src.model import MedicalCNNModel
        m = MedicalCNNModel(input_shape=(32, 32, 1), num_classes=3)
        m.build_model()
        m.compile_model(learning_rate=1e-3)

    def test_micro_train_completes(self):
        from src.model import MedicalCNNModel
        m = MedicalCNNModel(
            input_shape=(32, 32, 1),
            num_classes=3,
            model_save_path="models/micro_test.keras",
        )
        m.build_model()
        rng = np.random.default_rng(42)
        X = rng.random((12, 32, 32)).astype(np.float32)
        y = np.array([0, 1, 2] * 4, dtype=np.int32)
        history = m.train(X, y, epochs=2, batch_size=4, validation_split=0.2, patience=2)
        assert history is not None
        assert "loss" in history.history
