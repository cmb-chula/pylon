import contextlib

import torch.nn as nn


@contextlib.contextmanager
def set_mode(net: nn.Module, mode: str):
    """set the model's mode in a context manager"""
    before_train = net.training
    if mode == 'train':
        net.train()
    elif mode == 'eval':
        net.eval()
    else:
        raise NotImplementedError()

    yield net

    if before_train:
        net.train()
    else:
        net.eval()
