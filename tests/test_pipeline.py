"""
tests/test_pipeline.py
-----------------------
Pytest tests for Module 4: RealTimeProcessor end-to-end pipeline

Tests:
  - processing_pipeline returns AnomalyResult with all required fields
  - AnomalyResult fields are correctly typed and non-empty
  - Processing time < 500ms on CPU for a single synthetic image
  - batch_process returns correct number of results
  - get_statistics returns expected keys
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model():
    """Build a quick micro-trained MedicalCNNModel for pipeline testing."""
    from src.model import MedicalCNNModel

    m = MedicalCNNModel(
        input_shape=(64, 64, 1),
        num_classes=3,
        class_names=["normal", "tumor", "fracture"],
        model_save_path="models/pipeline_test.keras",
    )
    m.build_model()

    rng = np.random.default_rng(42)
    X = rng.random((18, 64, 64)).astype(np.float32)
    y = np.array([0, 1, 2] * 6, dtype=np.int32)
    m.train(X, y, epochs=2, batch_size=6, validation_split=0.2, patience=2)
    return m


@pytest.fixture(scope="module")
def processor(trained_model):
    from src.realtime_processor import RealTimeProcessor

    return RealTimeProcessor(
        model=trained_model,
        class_names=["normal", "tumor", "fracture"],
        image_size=(64, 64),
        queue_size=20,
        n_workers=2,
        anomaly_threshold=0.99,  # set high so no false alerts in tests
        alert_confidence=0.99,
    )


@pytest.fixture()
def synthetic_image() -> np.ndarray:
    rng = np.random.default_rng(77)
    return rng.random((64, 64)).astype(np.float32)


# ---------------------------------------------------------------------------
# AnomalyResult field completeness
# ---------------------------------------------------------------------------

class TestAnomalyResultFields:
    def test_returns_anomaly_result(self, processor, synthetic_image):
        from src.realtime_processor import AnomalyResult
        result = processor.processing_pipeline(synthetic_image)
        assert isinstance(result, AnomalyResult)

    def test_image_id_populated(self, processor, synthetic_image):
        result = processor.processing_pipeline(synthetic_image)
        assert result.image_id and len(result.image_id) > 0

    def test_timestamp_is_recent(self, processor, synthetic_image):
        result = processor.processing_pipeline(synthetic_image)
        assert result.timestamp > time.time() - 60

    def test_predicted_class_is_valid(self, processor, synthetic_image):
        result = processor.processing_pipeline(synthetic_image)
        assert result.predicted_class in ["normal", "tumor", "fracture"]

    def test_confidence_in_range(self, processor, synthetic_image):
        result = processor.processing_pipeline(synthetic_image)
        assert 0.0 <= result.confidence <= 1.0 + 1e-6

    def test_probabilities_sum_to_one(self, processor, synthetic_image):
        result = processor.processing_pipeline(synthetic_image)
        assert abs(result.probabilities.sum() - 1.0) < 1e-5

    def test_feature_vector_populated(self, processor, synthetic_image):
        result = processor.processing_pipeline(synthetic_image)
        assert isinstance(result.feature_vector, np.ndarray)
        assert result.feature_vector.ndim == 1
        assert len(result.feature_vector) > 0

    def test_anomaly_score_nonnegative(self, processor, synthetic_image):
        result = processor.processing_pipeline(synthetic_image)
        assert result.anomaly_score >= 0.0

    def test_processing_time_positive(self, processor, synthetic_image):
        result = processor.processing_pipeline(synthetic_image)
        assert result.processing_time_ms > 0.0


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestProcessingPerformance:
    def test_single_image_under_500ms(self, processor):
        rng = np.random.default_rng(42)
        img = rng.random((64, 64)).astype(np.float32)
        t0 = time.perf_counter()
        result = processor.processing_pipeline(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 500.0, (
            f"Processing time {elapsed_ms:.1f}ms exceeded 500ms threshold"
        )

    def test_throughput_at_least_5_per_sec(self, processor):
        """Run 10 images and check we hit ≥5 img/sec (conservative for CI)."""
        rng = np.random.default_rng(55)
        images = [rng.random((64, 64)).astype(np.float32) for _ in range(10)]
        t0 = time.perf_counter()
        for img in images:
            processor.processing_pipeline(img)
        elapsed = time.perf_counter() - t0
        throughput = len(images) / elapsed
        assert throughput >= 5.0, f"Throughput only {throughput:.1f} img/s (need ≥5)"


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

class TestBatchProcess:
    def test_batch_returns_correct_count(self, processor):
        rng = np.random.default_rng(21)
        images = [rng.random((64, 64)).astype(np.float32) for _ in range(5)]
        results = processor.batch_process(images, n_workers=2, show_progress=False)
        assert len(results) == 5

    def test_batch_all_results_typed(self, processor):
        from src.realtime_processor import AnomalyResult
        rng = np.random.default_rng(22)
        images = [rng.random((64, 64)).astype(np.float32) for _ in range(4)]
        results = processor.batch_process(images, show_progress=False)
        for r in results:
            assert isinstance(r, AnomalyResult)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestGetStatistics:
    def test_statistics_keys(self, processor, synthetic_image):
        processor.processing_pipeline(synthetic_image)
        stats = processor.get_statistics()
        expected_keys = {
            "n_processed",
            "throughput_per_sec",
            "avg_confidence",
            "avg_processing_time_ms",
            "class_distribution",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_n_processed_increments(self, processor):
        initial = processor.get_statistics()["n_processed"]
        rng = np.random.default_rng(88)
        processor.processing_pipeline(rng.random((64, 64)).astype(np.float32))
        updated = processor.get_statistics()["n_processed"]
        assert updated == initial + 1


# ---------------------------------------------------------------------------
# Compute Anomaly Score
# ---------------------------------------------------------------------------

class TestComputeAnomalyScore:
    def test_above_threshold_flagged(self, processor):
        # Set threshold low for this test
        processor.anomaly_threshold = 0.001
        assert processor.compute_anomaly_score(1.0) is True
        processor.anomaly_threshold = 0.99  # restore

    def test_below_threshold_not_flagged(self, processor):
        processor.anomaly_threshold = 0.5
        assert processor.compute_anomaly_score(0.001) is False
        processor.anomaly_threshold = 0.99  # restore
