"""
src/realtime_processor.py
--------------------------
Module 4: Real-time Image Processing Pipeline

Implements :class:`RealTimeProcessor` which ingests raw medical images,
chains the full preprocessing → feature extraction → CNN inference pipeline,
and streams :class:`AnomalyResult` objects to consumers via a thread-safe queue.

Key capabilities:
  - Folder-watching via ``watchdog``
  - Thread-pool parallel batch processing
  - Autoencoder-based anomaly scoring
  - Configurable alert threshold
  - Live throughput / latency statistics
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from src.preprocessing import MedicalImagePreprocessor
from src.feature_engineering import FeatureEngineer
from src.model import MedicalCNNModel
from src.utils import Timer, apply_colormap_to_heatmap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AnomalyResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """Container for the output of a single image's processing pipeline.

    Attributes:
        image_id: Unique identifier string (UUID-based).
        timestamp: Unix epoch float when inference completed.
        predicted_class: String class name (e.g. ``'tumor'``).
        predicted_index: Integer class index.
        confidence: Prediction probability for the predicted class (0–1).
        anomaly_score: Reconstruction error from autoencoder; higher = more anomalous.
        feature_vector: 1-D float32 feature array extracted from image.
        gradcam_overlay: uint8 RGB array with Grad-CAM overlay, or ``None``.
        processing_time_ms: Total wall-clock time for the pipeline in milliseconds.
        probabilities: Full softmax probability vector.
        is_alert: True if confidence exceeds alert threshold for a non-normal class.
        source_path: Original image file path if applicable.
    """

    image_id: str
    timestamp: float
    predicted_class: str
    predicted_index: int
    confidence: float
    anomaly_score: float
    feature_vector: np.ndarray
    gradcam_overlay: Optional[np.ndarray]
    processing_time_ms: float
    probabilities: np.ndarray
    is_alert: bool = False
    source_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Watchdog event handler (folder watch)
# ---------------------------------------------------------------------------

class _ImageEventHandler:
    """Watchdog FileCreatedEvent handler for new images in a watch folder."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".nii", ".gz", ".bmp", ".tiff"}

    def __init__(self, callback: Callable[[Path], None]) -> None:
        self._callback = callback

    def dispatch(self, event: Any) -> None:
        """Called by watchdog on any filesystem event."""
        try:
            if hasattr(event, "src_path"):
                p = Path(event.src_path)
                if p.suffix.lower() in self.SUPPORTED_EXTENSIONS and not event.is_directory:
                    self._callback(p)
        except Exception as exc:
            logger.warning("Event dispatch error: %s", exc)


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class RealTimeProcessor:
    """Orchestrates the full anomaly detection pipeline for real-time use.

    Args:
        model: Trained :class:`~src.model.MedicalCNNModel` instance.
        class_names: List of class name strings indexed by integer label.
        image_size: ``(H, W)`` target for preprocessing.
        queue_size: Maximum size of the internal input queue.
        n_workers: Worker threads for queue-based and batch processing.
        anomaly_threshold: Reconstruction error above which images are flagged
            as anomalous regardless of classifier output.
        alert_confidence: Minimum confidence for non-normal classifications
            to trigger an alert.
        normalization_method: ``'minmax'`` or ``'zscore'``, passed to preprocessor.
        random_seed: Seed for reproducibility in augmentation.
    """

    def __init__(
        self,
        model: MedicalCNNModel,
        class_names: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (128, 128),
        queue_size: int = 100,
        n_workers: int = 4,
        anomaly_threshold: float = 0.05,
        alert_confidence: float = 0.85,
        normalization_method: str = "minmax",
        random_seed: int = 42,
    ) -> None:
        self.model = model
        self.class_names = class_names or model.class_names
        self.image_size = image_size
        self.n_workers = n_workers
        self.anomaly_threshold = anomaly_threshold
        self.alert_confidence = alert_confidence

        # Pipeline components
        self.preprocessor = MedicalImagePreprocessor(
            image_size=image_size,
            normalization_method=normalization_method,
            augmentation_enabled=False,
            random_seed=random_seed,
        )
        self.feature_engineer = FeatureEngineer()

        # Thread-safe queues
        self._input_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._results_queue: queue.Queue = queue.Queue()

        # Runtime state
        self._worker_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._observer: Optional[Any] = None  # watchdog observer

        # Statistics accumulators
        self._stats_lock = threading.Lock()
        self._n_processed: int = 0
        self._total_time_ms: float = 0.0
        self._class_counts: Dict[str, int] = {cn: 0 for cn in self.class_names}
        self._confidence_sum: float = 0.0
        self._latencies: List[float] = []
        self._start_time: float = time.time()

        logger.info(
            "RealTimeProcessor ready | classes=%s | workers=%d | alert_conf=%.2f",
            self.class_names,
            n_workers,
            alert_confidence,
        )

    # ------------------------------------------------------------------
    # Image Ingestion
    # ------------------------------------------------------------------

    def ingest_image(self, image_path_or_array: Union[str, Path, np.ndarray]) -> str:
        """Submit an image (path or numpy array) to the processing queue.

        Args:
            image_path_or_array: File path to a medical image *or* a raw
                NumPy array of shape ``(H, W)`` or ``(H, W, C)``.

        Returns:
            Generated ``image_id`` string for tracking.

        Raises:
            queue.Full: If the input queue is at capacity.
        """
        image_id = str(uuid.uuid4())[:8]
        source_path: Optional[str] = None

        if isinstance(image_path_or_array, (str, Path)):
            source_path = str(image_path_or_array)
            item = {"id": image_id, "path": source_path, "array": None}
        else:
            item = {"id": image_id, "path": None, "array": image_path_or_array}

        self._input_queue.put_nowait(item)
        logger.debug("Ingested image %s | source=%s", image_id, source_path or "array")
        return image_id

    # ------------------------------------------------------------------
    # Single-image pipeline
    # ------------------------------------------------------------------

    def processing_pipeline(
        self,
        image: np.ndarray,
        image_id: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> AnomalyResult:
        """Run the full preprocessing → feature extraction → inference pipeline.

        Args:
            image: Raw float32 2-D image array (already loaded from disk).
            image_id: Optional identifier string.
            source_path: Optional originating file path string.

        Returns:
            Populated :class:`AnomalyResult` dataclass.
        """
        image_id = image_id or str(uuid.uuid4())[:8]
        t_start = time.perf_counter()

        # 1 — Preprocess
        preprocessed = self.preprocessor.preprocess_single(image)

        # 2 — Feature extraction
        try:
            feat_dict = {}
            feat_dict.update(self.feature_engineer.extract_texture_features(preprocessed))
            feat_dict.update(self.feature_engineer.extract_intensity_features(preprocessed))
            feat_dict.update(self.feature_engineer.extract_frequency_features(preprocessed))
            feature_vector = np.array(list(feat_dict.values()), dtype=np.float32)
        except Exception as exc:
            logger.warning("Feature extraction failed for %s: %s", image_id, exc)
            feature_vector = np.zeros(50, dtype=np.float32)

        # 3 — CNN inference
        probabilities = self.model.predict(preprocessed)
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        predicted_class = self.class_names[predicted_index] if predicted_index < len(self.class_names) else str(predicted_index)

        # 4 — Anomaly score (autoencoder reconstruction error)
        anomaly_score = 0.0
        if self.model.autoencoder is not None:
            try:
                errors = self.model.reconstruction_error(preprocessed[np.newaxis])
                anomaly_score = float(errors[0])
            except Exception as exc:
                logger.debug("Autoencoder scoring failed: %s", exc)

        # 5 — Grad-CAM overlay
        gradcam_overlay: Optional[np.ndarray] = None
        try:
            heatmap = self.model.get_gradcam(preprocessed, layer_name="conv4")
            gradcam_overlay = apply_colormap_to_heatmap(heatmap, preprocessed)
        except Exception as exc:
            logger.debug("Grad-CAM failed for %s: %s", image_id, exc)

        elapsed_ms = (time.perf_counter() - t_start) * 1_000

        # Flag alerts
        is_alert = (
            predicted_class != "normal"
            and confidence >= self.alert_confidence
        ) or (anomaly_score > self.anomaly_threshold)

        result = AnomalyResult(
            image_id=image_id,
            timestamp=time.time(),
            predicted_class=predicted_class,
            predicted_index=predicted_index,
            confidence=confidence,
            anomaly_score=anomaly_score,
            feature_vector=feature_vector,
            gradcam_overlay=gradcam_overlay,
            processing_time_ms=elapsed_ms,
            probabilities=probabilities,
            is_alert=is_alert,
            source_path=source_path,
        )

        self._update_statistics(result)
        if is_alert:
            self.alert(result)

        logger.debug(
            "Processed %s → %s (%.1f%%) in %.1f ms",
            image_id,
            predicted_class,
            confidence * 100,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Queue-based Worker Thread
    # ------------------------------------------------------------------

    def start_worker(self) -> None:
        """Start the background worker thread that drains the input queue."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("Worker thread already running.")
            return
        self._running.set()
        self._worker_thread = threading.Thread(
            target=self._process_queue_loop, daemon=True, name="RT-Worker"
        )
        self._worker_thread.start()
        logger.info("Worker thread started.")

    def stop_worker(self) -> None:
        """Signal the worker thread to stop after draining the queue."""
        self._running.clear()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
        logger.info("Worker thread stopped.")

    def _process_queue_loop(self) -> None:
        """Internal loop: dequeue items and run pipeline."""
        while self._running.is_set() or not self._input_queue.empty():
            try:
                item = self._input_queue.get(timeout=0.5)
                image = self._load_item(item)
                if image is None:
                    self._input_queue.task_done()
                    continue
                result = self.processing_pipeline(image, item["id"], item.get("path"))
                self._results_queue.put(result)
                self._input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error("Worker error: %s", exc)

    def process_queue(self) -> None:
        """Public alias for starting the queue-processing loop inline.

        Blocks until the ``_running`` event is cleared and queue is drained.
        """
        self.start_worker()

    def _load_item(self, item: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load an image from a queue item dict."""
        if item.get("array") is not None:
            return item["array"].astype(np.float32)
        if item.get("path"):
            try:
                return self.preprocessor.load_dicom(item["path"])
            except Exception as exc:
                logger.warning("Cannot load %s: %s", item["path"], exc)
        return None

    def get_results(self) -> Generator[AnomalyResult, None, None]:
        """Generator that yields completed :class:`AnomalyResult` objects.

        Yields:
            :class:`AnomalyResult` as they become available.

        Note:
            Blocks waiting for results.  Use with a timeout or stop condition.
        """
        while True:
            try:
                yield self._results_queue.get(timeout=1.0)
            except queue.Empty:
                return

    # ------------------------------------------------------------------
    # Folder Watching
    # ------------------------------------------------------------------

    def start_stream(
        self,
        input_folder: Union[str, Path],
        watch: bool = True,
        poll_interval: float = 0.5,
    ) -> None:
        """Watch *input_folder* for new image files, processing each automatically.

        Args:
            input_folder: Directory path to watch.
            watch: If ``True``, use ``watchdog`` for inotify-style watching.
                If ``False``, poll the directory periodically.
            poll_interval: Poll interval in seconds (only used when ``watch=False``).

        Raises:
            NotADirectoryError: If *input_folder* does not exist.
        """
        folder = Path(input_folder).resolve()
        if not folder.is_dir():
            raise NotADirectoryError(f"{input_folder} is not a directory.")

        self.start_worker()

        if watch:
            self._start_watchdog(folder)
        else:
            self._start_polling(folder, poll_interval)

    def _start_watchdog(self, folder: Path) -> None:
        """Use watchdog to react to filesystem events."""
        try:
            from watchdog.observers import Observer  # type: ignore
            from watchdog.events import FileSystemEventHandler  # type: ignore
        except ImportError:
            logger.warning("watchdog not installed; falling back to polling.")
            self._start_polling(folder, 1.0)
            return

        class _Handler(FileSystemEventHandler):
            def __init__(self, callback: Callable) -> None:
                self._cb = callback

            def on_created(self, event: Any) -> None:
                if not event.is_directory:
                    p = Path(event.src_path)
                    if p.suffix.lower() in MedicalImagePreprocessor.SUPPORTED_EXTENSIONS:
                        self._cb(p)

        handler = _Handler(callback=lambda p: self.ingest_image(p))
        self._observer = Observer()
        self._observer.schedule(handler, str(folder), recursive=False)
        self._observer.start()
        logger.info("Watchdog observer started on %s", folder)

    def _start_polling(self, folder: Path, interval: float) -> None:
        """Poll *folder* on a background thread for new image files."""
        seen: set = set()

        def _poll_loop() -> None:
            while self._running.is_set():
                for fp in folder.iterdir():
                    if fp.suffix.lower() in MedicalImagePreprocessor.SUPPORTED_EXTENSIONS and fp not in seen:
                        seen.add(fp)
                        self.ingest_image(fp)
                time.sleep(interval)

        t = threading.Thread(target=_poll_loop, daemon=True, name="RT-Poller")
        t.start()

    def stop_stream(self) -> None:
        """Stop the watchdog observer and worker thread."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
        self.stop_worker()
        logger.info("Stream stopped.")

    # ------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------

    def batch_process(
        self,
        image_list: List[Union[str, Path, np.ndarray]],
        n_workers: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[AnomalyResult]:
        """Parallel batch processing with ``ThreadPoolExecutor``.

        Args:
            image_list: List of file paths or NumPy arrays to process.
            n_workers: Number of parallel threads (defaults to ``self.n_workers``).
            show_progress: Print tqdm progress bar if available.

        Returns:
            List of :class:`AnomalyResult` in input order.
        """
        n_workers = n_workers or self.n_workers
        results: List[Optional[AnomalyResult]] = [None] * len(image_list)

        def _worker(idx_item: Tuple[int, Any]) -> Tuple[int, AnomalyResult]:
            idx, item = idx_item
            try:
                if isinstance(item, (str, Path)):
                    img = self.preprocessor.load_dicom(item)
                    src = str(item)
                else:
                    img = item.astype(np.float32)
                    src = None
                res = self.processing_pipeline(img, source_path=src)
            except Exception as exc:
                logger.warning("Batch item %d failed: %s", idx, exc)
                # Return a dummy result
                res = AnomalyResult(
                    image_id=f"err_{idx}",
                    timestamp=time.time(),
                    predicted_class="error",
                    predicted_index=-1,
                    confidence=0.0,
                    anomaly_score=0.0,
                    feature_vector=np.zeros(1, dtype=np.float32),
                    gradcam_overlay=None,
                    processing_time_ms=0.0,
                    probabilities=np.zeros(len(self.class_names), dtype=np.float32),
                )
            return idx, res

        indexed = list(enumerate(image_list))

        try:
            from tqdm import tqdm  # type: ignore
            ctx: Any = tqdm(total=len(indexed), desc="Batch processing", unit="img") if show_progress else None
        except ImportError:
            ctx = None

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker, item): item[0] for item in indexed}
            for fut in as_completed(futures):
                idx, res = fut.result()
                results[idx] = res
                if ctx is not None:
                    ctx.update(1)

        if ctx is not None:
            ctx.close()

        logger.info("Batch processing complete: %d images", len(results))
        return [r for r in results if r is not None]  # type: ignore

    # ------------------------------------------------------------------
    # Anomaly Scoring
    # ------------------------------------------------------------------

    def compute_anomaly_score(self, reconstruction_error: float) -> bool:
        """Determine whether a reconstruction error indicates an anomaly.

        Args:
            reconstruction_error: Scalar MSE value from autoencoder.

        Returns:
            ``True`` if the image should be flagged as anomalous.
        """
        is_anomalous = reconstruction_error > self.anomaly_threshold
        logger.debug(
            "Anomaly score %.4f vs threshold %.4f → %s",
            reconstruction_error,
            self.anomaly_threshold,
            "ANOMALY" if is_anomalous else "NORMAL",
        )
        return is_anomalous

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    def alert(self, result: AnomalyResult) -> None:
        """Log and print an alert for high-confidence anomaly detections.

        Args:
            result: The :class:`AnomalyResult` triggering the alert.
        """
        msg = (
            f"🚨 ANOMALY ALERT | id={result.image_id} | "
            f"class={result.predicted_class.upper()} | "
            f"conf={result.confidence:.1%} | "
            f"score={result.anomaly_score:.4f} | "
            f"time={result.processing_time_ms:.1f}ms"
        )
        logger.warning(msg)
        print(msg)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _update_statistics(self, result: AnomalyResult) -> None:
        """Thread-safe accumulation of runtime statistics."""
        with self._stats_lock:
            self._n_processed += 1
            self._total_time_ms += result.processing_time_ms
            self._confidence_sum += result.confidence
            self._latencies.append(result.processing_time_ms)
            cls = result.predicted_class
            if cls in self._class_counts:
                self._class_counts[cls] += 1
            else:
                self._class_counts[cls] = 1

    def get_statistics(self) -> Dict[str, Any]:
        """Return a snapshot of running pipeline statistics.

        Returns:
            Dictionary with keys:
                - ``n_processed`` – total images processed.
                - ``throughput_per_sec`` – images / second since start.
                - ``avg_confidence`` – mean prediction confidence.
                - ``avg_processing_time_ms`` – mean pipeline latency.
                - ``class_distribution`` – per-class sample counts.
                - ``latency_p50_ms`` – median latency.
                - ``latency_p95_ms`` – 95th percentile latency.
                - ``latency_p99_ms`` – 99th percentile latency.
        """
        with self._stats_lock:
            n = self._n_processed
            elapsed = max(time.time() - self._start_time, 1e-3)
            latencies = np.array(self._latencies, dtype=np.float32)

            return {
                "n_processed": n,
                "throughput_per_sec": n / elapsed,
                "avg_confidence": self._confidence_sum / n if n > 0 else 0.0,
                "avg_processing_time_ms": self._total_time_ms / n if n > 0 else 0.0,
                "class_distribution": dict(self._class_counts),
                "latency_p50_ms": float(np.percentile(latencies, 50)) if len(latencies) > 0 else 0.0,
                "latency_p95_ms": float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0.0,
                "latency_p99_ms": float(np.percentile(latencies, 99)) if len(latencies) > 0 else 0.0,
            }
