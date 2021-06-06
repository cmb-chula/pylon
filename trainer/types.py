"""
Define data types

from: https://github.com/fastai/fastai/blob/master/fastai/torch_core.py
"""

import collections
from functools import partial
from typing import (Any, AnyStr, Callable, Collection, Dict, Hashable,
                    Iterator, List, Mapping, NewType, Optional, Sequence,
                    Tuple, TypeVar, Union)

import numpy as np
from torch import (ByteTensor, DoubleTensor, FloatTensor, HalfTensor,
                   LongTensor, ShortTensor, Tensor)

NPArray = np.ndarray
LambdaFn = Callable[[Tensor], Tensor]


def is_python_primitive(x):
    """whether the object is python primitive"""
    return isinstance(x, (type(None), int, float, bool, str))


def is_numpy(x):
    return isinstance(x, np.ndarray)


def detach(t: Tensor):
    if is_python_primitive(t) or is_numpy(t):
        return t
    elif isinstance(t, Tensor):
        return t.detach()
    else:
        return apply_structure(t, detach)


def detach_(t: Tensor):
    if is_python_primitive(t) or is_numpy(t):
        return t
    elif isinstance(t, Tensor):
        return t.detach_()
    else:
        return apply_structure(t, detach_)


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


def cuda(t: Tensor, dev='cuda:0'):
    if is_python_primitive(t) or is_numpy(t):
        return t
    elif isinstance(t, Tensor):
        return t.to(dev)
    else:
        return apply_structure(t, partial(cuda, dev=dev))


def apply_structure(structure, fn, max_depth=2, current_depth=0):
    """Apply fn onto a structure and return the transformed structure

    Args:
        structure: list, dict, singleton
        fn: a transformation function
    """
    # don't do anything beyond certain level
    if current_depth == max_depth:
        return structure

    if isinstance(structure, dict):
        # support dict
        return {
            key: apply_structure(structure[key],
                                 fn,
                                 max_depth=max_depth,
                                 current_depth=current_depth + 1)
            for key in structure
        }
    elif isinstance(structure, (tuple, list)):
        # support list and tuples
        _out = [
            apply_structure(e,
                            fn,
                            max_depth=max_depth,
                            current_depth=current_depth + 1) for e in structure
        ]
        if isinstance(structure, tuple):
            _out = tuple(_out)
        return _out
    else:
        # NOTE: single object
        return fn(structure)
