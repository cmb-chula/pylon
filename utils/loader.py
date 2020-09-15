from torch import Tensor
from torch.utils.data import DataLoader

from utils.structure import *


class BaseLoaderWrapper:
    def __init__(self, loader: DataLoader):
        self.loader = loader

    def stats(self):
        if isinstance(self.loader, BaseLoaderWrapper):
            return self.loader.stats()
        else:
            return {}

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.loader)

    def __repr__(self):
        return str(self)


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
