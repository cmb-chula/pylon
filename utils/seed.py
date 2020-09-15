import random

import numpy as np
import torch


def set_seed(seed: int):
    """ Set random seed for python, numpy and pytorch RNGs """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_random_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'pytorch': torch.get_rng_state(),
    }


def set_random_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['pytorch'])
