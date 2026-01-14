"""
Training utilities module for JAxtar.

Provides dataset generation, optimizer factories, and loss functions
for neural heuristic and Q-function training.
"""

from train_util import losses, optimizer, sampling

__all__ = ["optimizer", "sampling", "losses"]
