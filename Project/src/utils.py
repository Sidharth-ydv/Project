"""
src/utils.py
-----------
Shared utility functions and helpers used across all modules.

Provides:
    - Logging setup
    - Config loading / merging
    - Directory management
    - Timing utilities
    - Image I/O helpers
    - Color map utilities for Grad-CAM overlays
"""

from __future__ import annotations

import logging
import os
import sys
import time
import functools
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Configure root logger with console + optional file handler.

    Args:
        log_level: Python logging level string ('DEBUG', 'INFO', etc.).
        log_file: Optional path to write log file. Directory is created if
            it does not exist.
        log_format: Format string for log messages.
        datefmt: Date/time format for log messages.

    Returns:
        Configured root :class:`logging.Logger` instance.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        ensure_dir(str(Path(log_file).parent))
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
    return logging.getLogger()


def get_logger(name: str) -> logging.Logger:
    """Return a named module-level logger.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file and return as nested dict.

    Args:
        config_path: Absolute or relative path to ``config.yaml``.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        yaml.YAMLError: On YAML parse errors.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)

    return cfg


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge *override* into *base* config dict.

    Args:
        base: Base configuration dictionary.
        override: Override values (only keys present here are updated).

    Returns:
        Merged configuration dictionary (new object, inputs unchanged).
    """
    merged: Dict[str, Any] = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = merge_configs(merged[key], val)
        else:
            merged[key] = val
    return merged


# ---------------------------------------------------------------------------
# File System
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and all parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        Resolved :class:`pathlib.Path` object.
    """
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(path: Union[str, Path], base: Optional[Union[str, Path]] = None) -> Path:
    """Resolve *path* relative to *base* (cwd if *base* is None).

    Args:
        path: Absolute or relative path string / Path object.
        base: Base directory for resolution.

    Returns:
        Resolved absolute :class:`pathlib.Path`.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    if base is not None:
        return (Path(base) / p).resolve()
    return p.resolve()


# ---------------------------------------------------------------------------
# Timing / Performance
# ---------------------------------------------------------------------------

class Timer:
    """Context-manager that measures wall-clock elapsed time in milliseconds.

    Example::

        with Timer() as t:
            do_work()
        print(t.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1_000


def timeit(func: Callable) -> Callable:
    """Decorator that logs execution time of *func* at DEBUG level.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function with identical signature.
    """
    logger = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1_000
        logger.debug("%s completed in %.2f ms", func.__qualname__, elapsed)
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Image Utilities
# ---------------------------------------------------------------------------

def to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale a float image array to ``uint8`` (0–255).

    Args:
        image: Float array with values in any range.

    Returns:
        ``uint8`` NumPy array clamped to [0, 255].
    """
    img = image.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - lo) / (hi - lo) * 255.0
    return img.clip(0, 255).astype(np.uint8)


def apply_colormap_to_heatmap(
    heatmap: np.ndarray,
    original: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a Grad-CAM/activation heatmap on an original image.

    Args:
        heatmap: 2-D float array in [0, 1]; the activation map.
        original: 2-D or 3-D ``uint8`` array of the original image.
        alpha: Blend weight for the heatmap (0 = original only, 1 = heatmap only).

    Returns:
        3-D ``uint8`` RGB array with heatmap blended in.
    """
    import cv2  # local import to avoid top-level hard dependency at module load

    heatmap_uint8 = to_uint8(heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    if original.ndim == 2:
        orig_rgb = cv2.cvtColor(to_uint8(original), cv2.COLOR_GRAY2RGB)
    else:
        orig_rgb = to_uint8(original)

    # Resize colored heatmap to match original
    if colored_rgb.shape[:2] != orig_rgb.shape[:2]:
        colored_rgb = cv2.resize(
            colored_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    overlay = (alpha * colored_rgb.astype(np.float32) + (1 - alpha) * orig_rgb.astype(np.float32))
    return overlay.clip(0, 255).astype(np.uint8)


def pad_or_crop_to_square(image: np.ndarray) -> np.ndarray:
    """Pad a non-square image with zeros to make it square.

    Args:
        image: 2-D or 3-D NumPy array.

    Returns:
        Square NumPy array with same dtype as *image*.
    """
    h, w = image.shape[:2]
    side = max(h, w)
    if image.ndim == 2:
        padded = np.zeros((side, side), dtype=image.dtype)
        padded[:h, :w] = image
    else:
        padded = np.zeros((side, side, image.shape[2]), dtype=image.dtype)
        padded[:h, :w] = image
    return padded


# ---------------------------------------------------------------------------
# Misc Helpers
# ---------------------------------------------------------------------------

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer label array to one-hot encoded matrix.

    Args:
        labels: 1-D integer array of class indices.
        num_classes: Total number of classes.

    Returns:
        2-D float32 array of shape ``(len(labels), num_classes)``.
    """
    encoded = np.zeros((len(labels), num_classes), dtype=np.float32)
    encoded[np.arange(len(labels)), labels.astype(int)] = 1.0
    return encoded


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis.

    Args:
        x: Input array of arbitrary shape.

    Returns:
        Array of same shape with softmax applied along last axis.
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def class_distribution(labels: np.ndarray, class_names: Optional[list] = None) -> Dict[str, int]:
    """Compute per-class sample count dictionary.

    Args:
        labels: Integer label array.
        class_names: Optional list of class name strings indexed by label integer.

    Returns:
        Dictionary mapping class name (or integer string) → sample count.
    """
    unique, counts = np.unique(labels, return_counts=True)
    dist: Dict[str, int] = {}
    for u, c in zip(unique, counts):
        key = class_names[int(u)] if class_names else str(int(u))
        dist[key] = int(c)
    return dist
