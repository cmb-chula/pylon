import os
from collections import defaultdict
from functools import partial
from typing import List

import pandas as pd
from torch import Tensor

from ..save import *
from ..stateful import Stateful


class Callback(Stateful):
    """
    when not to use callbacks:
    - if it is required for correct forward pass of a model, that should be in trainer
    """
    _order = 100

    def save(self, path: str):
        """save state to file"""
        if self.is_state_empty():
            # don't need to save empty state
            return
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch_save(self.get_state(), path)

    def load(self, path: str, map_location=None):
        """load state from file"""
        if self.is_state_empty():
            # this cb doesn't need a state
            # caution: cb that needs a state must have the "footprint"
            # of the states, so that it would not be empty at first!
            # unless it will not be loaded!
            return
        self.load_state(torch.load(path, map_location=map_location))

    def on_train_begin(self, **kwargs):
        pass

    def on_ep_begin(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        pass

    def on_forward_begin(self, **kwargs):
        pass

    def on_forward_end(self, **kwargs):
        pass

    def on_backward_begin(self, **kwargs):
        pass

    def on_backward_end(self, **kwargs):
        pass

    def on_step_begin(self, **kwargs):
        pass

    def on_step_end(self, **kwargs):
        pass

    def on_batch_end(self, **kwargs):
        pass

    def on_ep_end(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_abrupt_end(self, **kwargs):
        pass

    def __str__(self):
        return self.__repr__()


class StatsCallback(Callback):
    """
    base class for callbacks that keep stats as "history" 
    """
    def __init__(self, n_log_cycle=1):
        super().__init__()
        self.n_log_cycle = n_log_cycle
        self._state['hist'] = defaultdict(list)
        # to be put to the progress bar (only)
        self.stats = {}
        # to be put to the history
        # we have buffer so that, we can update many times per iteration
        # not need to collect everything first
        self.buffer = {}

    @classmethod
    def collect_latest(cls, callbacks):
        """collect latest entries from all callbacks (that is StatsCallback), excluding i_itr"""
        out = {}
        for cb in callbacks:
            if isinstance(cb, StatsCallback):
                for k, v in cb.last_hist.items():
                    if k != 'i_itr':
                        out[k] = v
        return out

    @classmethod
    def combine_callbacks(cls, callbacks):
        """merge dataframes from callbacks"""
        out = None
        for cb in callbacks:
            if isinstance(cb, StatsCallback):
                df = cb.df
                if 'i_itr' in df:
                    if out is None:
                        out = cb.df
                    else:
                        out = pd.merge(out, cb.df, on='i_itr', how='outer')
        return out

    @property
    def hist(self):
        return self._state['hist']

    @property
    def last_hist(self):
        return {k: v[-1] for k, v in self.hist.items()}

    @property
    def df(self):
        return pd.DataFrame(self.hist)

    def on_batch_begin(self, **kwargs):
        # clear the buffer
        self.buffer = {}

    def on_batch_end(self, **kwargs):
        """auto-flush after each iteration"""
        self._flush()

    def progress_bar(self, d):
        """update the stats which shall be shown in the progress bar (only)"""
        assert 'i_itr' in d
        i_itr = d['i_itr']
        if self.is_log_cycle(i_itr):
            d = self._eval(d)
            self.stats.update(_strip(d))

    def bar_and_hist(self, d):
        """both update the progress bar and write to the buffer (history), don't forget to flush"""
        assert 'i_itr' in d
        i_itr = d['i_itr']
        if self.is_log_cycle(i_itr):
            d = self._eval(d)
            self.stats.update(_strip(d))
            self.buffer.update(_strip(d))

    def add_hist(self, d):
        """buffer before putting into the history after flushing"""
        assert 'i_itr' in d
        i_itr = d['i_itr']
        if self.is_log_cycle(i_itr):
            d = self._eval(d)
            self.buffer.update(_strip(d))

    def _flush(self):
        """save the buffer to history"""
        d = self.buffer
        if len(d) > 0:
            assert 'i_itr' in d, f'i_itr is not present in {self}'
            _append_dict(self.hist, d)
            # should not clear the buffer,
            # it might be used by others

    def is_log_cycle(self, i_itr):
        return i_itr % self.n_log_cycle == 0

    def _eval(self, d):
        for k, v in d.items():
            d[k] = _get_val(v)
        return d


def _strip(x):
    """remvoe tensor-hood from the input structure"""
    if isinstance(x, Tensor):
        x = x.item()
    elif isinstance(x, dict):
        x = {k: _strip(v) for k, v in x.items()}
    return x


def _get_val(v):
    """get val from a function or a value"""
    if callable(v):
        return v()
    return v


def _merge_df(dfs):
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = pd.merge(df, dfs[i], on='i_itr', how='outer')
    return df


def _append_dict(dict_of_list, dict):
    """
    append a dict into a dict of lists
    before doing that, all lists should have the same size first!
    append None to the smaller lists.
    """
    def fill_na():
        lengths = [len(v) for v in dict_of_list.values()]
        if len(lengths) == 0:
            max_len = 0
        else:
            max_len = max(lengths)

        # equate the dict sizes with None
        for k in dict_of_list:
            while len(dict_of_list[k]) < max_len:
                dict_of_list[k].append(None)

    fill_na()
    for k, v in dict.items():
        dict_of_list[k].append(v)
    fill_na()


def callback_call(callbacks: List[Callback], method: str, kwargs):
    """call a list of callbacks"""
    if callbacks is None:
        return
    if not isinstance(callbacks, list):
        callbacks = [callbacks]
    # ignore None callbacks
    callbacks = [cb for cb in callbacks if cb is not None]

    # the final return is the "OR" of all return values
    out = None
    for cb in sorted(callbacks, key=partial(_get_cb_order, meth=method)):
        fn = getattr(cb, method, None)
        assert fn is not None, f'the callback {cb} does not have {method}'
        if fn is not None:
            try:
                res = fn(**kwargs)
                assert res is None or isinstance(
                    res, bool
                ), f'returns from the callback {cb} must be either None or a boolean'
            except TypeError as e:
                print(f'type error: {e} ... at {cb}')
                raise e
            except Exception as e:
                print(f'error {e} ... at {cb}')
                raise e

            if res is not None:
                if out is None: out = res
                out |= res
    return out


def set_order(order):
    """decorator to set callback's method order
    usage: 
        @set_order(100)
        def method(self):
            pass
    """
    def inner(meth):
        def fn(*args, **kwargs):
            return meth(*args, **kwargs)

        fn._order = order
        return fn

    return inner


def _get_cb_order(cb, meth):
    fn = getattr(cb, meth, None)
    if fn is None:
        return cb._order
    # return the method's order (if not use the cb's order)
    order = getattr(fn, '_order', cb._order)
    return order


if __name__ == "__main__":
    a = StatsCallback()
    a.bar_and_hist({'a': 10, 'i_itr': 1})
    print(a.stats)
