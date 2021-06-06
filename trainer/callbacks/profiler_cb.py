import time
from contextlib import contextmanager

from .base_cb import *

_BUCKET = {}


def put_to_profiler(d: Dict):
    """update values to the bucket"""
    if d['i_itr'] is None:
        del d['i_itr']
    _BUCKET.update(d)


@contextmanager
def time_elapsed_to_profiler(key,
                             i_itr=None,
                             enable=True,
                             prefix='profiler/',
                             verbose=False):
    """a contextmanager that logs the time of the block
    Args:
        i_itr: only required once per iteration
    """
    begin_time = time.time()
    yield
    end_time = time.time()
    elapsed = end_time - begin_time
    if enable:
        put_to_profiler({
            'i_itr': i_itr,
            f'{prefix}{key}': elapsed,
        })
        if verbose:
            print(f'{prefix}{key}:', elapsed)


class ProfilerCb(BoardCallback):
    """
    this callback will retrieve the values from the bucket and put the tensorboard
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_batch_end(self, i_itr, **kwargs):
        global _BUCKET
        self.add_to_hist(_BUCKET)
        # clear bucket
        _BUCKET = {}
        super().on_batch_end(i_itr=i_itr, **kwargs)
