import math
import numpy as np
import matplotlib.pyplot as plt
from .rename import *


def const(val):
    @rename(f'const({val})')
    def fn(**kwargs):
        return val

    return fn


def linear(start, end):
    @rename(f'linear({start},{end})')
    def fn(p, **kwargs):
        return start + (end - start) * p

    return fn


def cos(start, end):
    @rename(f'cos({start},{end})')
    def fn(p, **kwargs):
        return start + (1 + math.cos(math.pi * (1 - p))) * (end - start) / 2

    return fn


def exp(start, end):
    @rename(f'exp({start},{end})')
    def fn(p, **kwargs):
        return start * (end / start)**p

    return fn


def decay_exp_epoch(rate_per_epoch):
    @rename(f'exp({rate_per_epoch})')
    def fn(f_ep, **kwargs):
        return rate_per_epoch**f_ep

    return fn


def warmup_linear(n_itr=None, n_ep=None):
    """warm up the learning rate from 0 to the optimizer's learning rate in n_itr or n_ep.
    """
    assert n_itr is not None or n_ep is not None, f'both n_itr and n_ep cannot be None at the same time'
    if n_itr is not None:
        name = f'warmup({n_itr}itr)'
    elif n_ep is not None:
        name = f'warmup({n_ep}ep)'
    else:
        raise NotImplementedError()

    @rename(name)
    def fn(i=None, f_ep=None, **kwargs):
        if n_itr is not None:
            if i <= n_itr:
                return i / n_itr
        elif n_ep is not None:
            if f_ep <= n_ep:
                return f_ep / n_ep

        return None

    return fn


def decay_linear(start_n_itr=None, start_n_ep=None):
    """decay the learning rate to zero"""
    assert start_n_itr is not None or start_n_ep is not None, f'both cannot be None at the same time'
    if start_n_itr is not None:
        name = f'decay({start_n_itr}itr)'
    elif start_n_ep is not None:
        name = f'decay({start_n_ep}ep)'
    else:
        raise NotImplementedError()

    @rename(name)
    def fn(i=None, f_ep=None, n_max_itr=None, n_max_ep=None, **kwargs):
        if start_n_itr is not None:
            if i >= start_n_itr:
                length = n_max_itr - start_n_itr
                return 1 - (i - start_n_itr) / length
        elif start_n_ep is not None:
            if f_ep >= start_n_ep:
                length = n_max_ep - start_n_ep
                return 1 - (f_ep - start_n_ep) / length

        return None

    return fn


def plot_lr(fn):
    space = np.linspace(0, 1, 1000)
    lrs = []
    for p in space:
        lrs.append(fn(p, None))
    plt.plot(space, lrs)


if __name__ == "__main__":
    pass