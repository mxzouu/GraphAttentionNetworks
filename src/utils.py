"""
src/utils.py
============

Petites fonctions utilitaires : fixation des seeds, accuracy, helpers d'affichage.
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fixe les seeds pour la reproductibilité (Python, NumPy, PyTorch CPU et CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Accuracy de classification.

    Args:
        logits: (N, C) log-probabilités ou logits (l'argmax est invariant).
        labels: (N,) labels entiers.

    Returns:
        accuracy en proportion (entre 0.0 et 1.0).
    """
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float().sum().item()
    return correct / labels.size(0)


def format_mean_std(values: list[float], scale: float = 100.0, digits: int = 1) -> str:
    """Formate une liste de valeurs en 'mean ± std' (en pourcentage par défaut)."""
    arr = np.array(values) * scale
    return f"{arr.mean():.{digits}f} ± {arr.std():.{digits}f}"
