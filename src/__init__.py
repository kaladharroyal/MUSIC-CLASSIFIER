"""
Research Extension Package: Interpretable Music Auto-Tagging & Genre Classification
IEEE Access & ICASSP Research Pipeline
"""

from .essentia_features import extract_essentia_signal_features, FEATURE_NAMES
from .midlevel_vggish import MidLevelVGGish, get_midlevel_feature_names
from .faithfulness_eval import evaluate_shap_faithfulness, plot_faithfulness_curves

__all__ = [
    "extract_essentia_signal_features",
    "FEATURE_NAMES",
    "MidLevelVGGish",
    "get_midlevel_feature_names",
    "evaluate_shap_faithfulness",
    "plot_faithfulness_curves"
]
