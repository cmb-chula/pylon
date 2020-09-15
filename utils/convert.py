import numpy as np
from torch import Tensor

from utils.structure import *


def is_python_primitive(x):
    """whether the object is python primitive"""
    return isinstance(x, (type(None), int, float, bool))


def is_numpy(x):
    return isinstance(x, np.ndarray)


def item(x: Tensor):
    if isinstance(x, Tensor): return x.item()
    else: return x


def cpu(t: Tensor):
    if is_python_primitive(t) or is_numpy(t):
        return t
    elif isinstance(t, Tensor):
        return t.cpu()
    else:
        return apply_structure(t, cpu)


def detach(t: Tensor):
    if is_python_primitive(t) or is_numpy(t):
        return t
    elif isinstance(t, Tensor):
        return t.detach()
    else:
        return apply_structure(t, detach)