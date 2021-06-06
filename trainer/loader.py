import torch

from .loader_base import *
from .prefetcher import *
from .types import *

class Infinite(BaseLoaderWrapper):
    def __iter__(self):
        while True:
            for x in self.loader:
                yield x

class FlattenLoader(BaseLoaderWrapper):
    """flatten the input data"""
    def __init__(self, loader):
        super().__init__(loader)

    def __iter__(self):
        for data in self.loader:
            x = data[0]
            x = x.reshape(x.shape[0], -1)
            yield (x, *data[1:])

class LambdaLoader(BaseLoaderWrapper):
    """apply a function over the loader"""
    def __init__(self, loader, fn):
        super().__init__(loader)
        self.fn = fn

    def __iter__(self):
        for data in self.loader:
            yield self.fn(data)

class ConvertLoader(BaseLoaderWrapper):
    """put the loader data to a device"""
    def __init__(self, loader, device):
        super().__init__(loader)
        self.device = device

    def __iter__(self):
        def conv(x):
            # apply only to the pytorch tensor
            if isinstance(x, Tensor):
                return x.to(self.device)
            return x

        for x in self.loader:
            yield apply_structure(x, conv)

class OneHotLoader(BaseLoaderWrapper):
    """turn a classification loader into one-hot loader
    """
    def __init__(self, loader, n_cls):
        super().__init__(loader)
        self.n_cls = n_cls

    def __iter__(self):
        for x, y in self.loader:
            yield x, one_hot(y, n_cls=self.n_cls)

def one_hot(y, n_cls):
    """turn an int based label into one-hot"""
    assert torch.max(y).item() < n_cls
    if y.dim() == 1:
        y = y.unsqueeze(1)
    onehot = torch.zeros(y.shape[0], n_cls).to(y.device)
    onehot.scatter_(1, y, 1)
    return onehot

class Truncate(BaseLoaderWrapper):
    """make a smaller dataset"""
    def __init__(self, loader, n_truncate, n_bs):
        super().__init__(loader)
        self.n_truncate = n_truncate
        self.n_bs = n_bs

    def __iter__(self):
        i = 0
        for data in self.loader:
            i += self.n_bs
            if i >= self.n_truncate:
                break
            yield data

class CacheLoader(BaseLoaderWrapper):
    """useful for small datasets"""
    def __init__(self, loader):
        super().__init__(loader)
        self.cache = None

    def __iter__(self):
        if self.cache is None:
            cache = []
            for data in self.loader:
                cache.append(data)
                yield data
            self.cache = cache
        else:
            for data in self.cache:
                yield data
