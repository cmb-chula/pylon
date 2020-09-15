from collections import deque

import numpy as np
from torch import Tensor


class Average:
    """True average without number overflow"""
    def __init__(self):
        self.n = 0
        self.avg = 0.

    def update(self, x, w: int = 1):
        self.n += w
        self.avg += w / self.n * (x - self.avg)

    def val(self):
        return self.avg


class SMA(Average):
    """Simple moving average
    note: not efficient for large size
    """
    def __init__(self, size=100):
        self.vals = deque(maxlen=size)
        self.weights = deque(maxlen=size)

    def update(self, x, w: int = 1):
        self.vals.append(x)
        self.weights.append(w)

    def val(self) -> Tensor:
        v, w = np.array(self.vals), np.array(self.weights)
        return (v * w).sum() / (w.sum() + 1e-8)
