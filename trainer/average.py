from collections import deque
from .types import *

class Average:
    """True average without number overflow"""
    def __init__(self):
        self.n = 0
        self.avg = 0.

    def update(self, x: Tensor, w: int = 1):
        self.n += w
        self.avg += w / self.n * (item(x) - self.avg)

    def val(self):
        return self.avg

class SMA(Average):
    """Simple moving average
    note: not efficient for large size
    """
    def __init__(self, size=100):
        self.vals = deque(maxlen=size)
        self.weights = deque(maxlen=size)

    def update(self, x: Tensor, w: int = 1):
        self.vals.append(item(x))
        self.weights.append(w)

    def val(self) -> Tensor:
        v, w = np.array(self.vals), np.array(self.weights)
        return (v * w).sum() / (w.sum() + 1e-8)

class ExponentialAverage(Average):
    """avg = beta * avg + (1 - beta) * x
    with correction early values
    """
    def __init__(self, beta=0.9, correction: bool = True):
        self.beta = beta
        self.correction = correction
        self.current = 0
        self.n = 0

    def update(self, x: Tensor):
        self.n += 1
        self.current = self.beta * self.current + (1 - self.beta) * item(x)

    def val(self):
        if self.correction:
            return self.current / (1 - self.beta**self.n)  # correction term
        else:
            return self.current

class HarmonicMean(Average):
    """used for rates averages"""
    def __init__(self, gain: float, correction: bool = False):
        self.gain = gain
        self.correction = correction
        self.inner = 0
        self.n = 0

    def update(self, x: Tensor):
        self.n += 1
        self.inner = (1 - self.gain) * self.inner + self.gain / (item(x) + 1e-7)

    def val(self):
        if self.n == 0:
            return self.inner
        if self.correction:
            corr = self.inner / (1 - (1 - self.gain)**self.n)
            return 1 / corr
        else:
            return 1 / self.inner
