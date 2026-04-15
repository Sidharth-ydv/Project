"""
medical_anomaly_detection/src/__init__.py
Package initializer — exposes key classes for cleaner imports.
"""

from src.preprocessing import MedicalImagePreprocessor
from src.feature_engineering import FeatureEngineer
from src.model import MedicalCNNModel
from src.realtime_processor import RealTimeProcessor, AnomalyResult
from src.evaluator import ModelEvaluator
from src.utils import setup_logging, load_config, ensure_dir

__all__ = [
    "MedicalImagePreprocessor",
    "FeatureEngineer",
    "MedicalCNNModel",
    "RealTimeProcessor",
    "AnomalyResult",
    "ModelEvaluator",
    "setup_logging",
    "load_config",
    "ensure_dir",
]
