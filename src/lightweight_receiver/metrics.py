"""Bit/block error rates and model size."""
import numpy as np


def bit_error_rate(predicted: np.ndarray, target: np.ndarray) -> float:
    """Fraction of incorrect bits."""
    return float(np.mean(np.asarray(predicted) != np.asarray(target)))


def block_error_rate(predicted: np.ndarray, target: np.ndarray) -> float:
    """Fraction of frames containing at least one incorrect information bit."""
    return float(np.mean(np.any(np.asarray(predicted) != np.asarray(target), axis=1)))


def parameter_count(model, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)
