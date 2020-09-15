from torch.utils.data import DataLoader

from ..base_predictor import BasePredictor
from .base_cb import *
from .report_cb import ProgressCb


class ValidateCb(StatsCallback):
    """validate every n iteration,
    it will not report anything by default

    To report loss and accuracy, use:
        callbacks=[AvgCb(['loss', 'acc])] 
    
    Args:
        n_val_cycle: default: n_ep_itr
        collect_keys: keys to be collected for the whole dataset to be used for calculating statistics
        on_end: extra validation before ending (even it doesn't divide)
    """
    def __init__(self,
                 loader: DataLoader,
                 n_val_cycle: int = None,
                 name: str = 'val',
                 callbacks=None,
                 on_end=False):
        # n_log_cycle = 1, when it say writes it should write
        super().__init__()
        self.loader = loader
        self.n_val_cycle = n_val_cycle
        self.name = name
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks + self.make_default_callbacks()
        self.on_end = on_end

    def make_default_callbacks(self):
        return [
            ProgressCb(self.name),
        ]

    def on_train_begin(self, n_ep_itr, **kwargs):
        if self.n_val_cycle is None:
            # set deafult to 1 epoch
            self.n_val_cycle = n_ep_itr

    # should run a bit early
    # so that others that might be able to use 'val_loss'
    @set_order(90)
    def on_batch_end(self, trainer, i_itr: int, n_max_itr: int, **kwargs):
        if ((self.on_end and i_itr == n_max_itr)
                or i_itr % self.n_val_cycle == 0):

            # make prediction
            predictor = BasePredictor(trainer,
                                      callbacks=self.callbacks,
                                      collect_keys=[])
            predictor.predict(self.loader)

            # collect all data from callbacks
            out = StatsCallback.collect_latest(self.callbacks)

            # the keys in the callbacks should be visible on the progress bar
            # everything else is kept in the buffer (not visible)
            bar_keys = set()
            for cb in self.callbacks:
                if isinstance(cb, StatsCallback):
                    bar_keys |= set(cb.stats.keys())
            bar_keys -= set(['i_itr'])

            bar = {'i_itr': i_itr}
            info = {'i_itr': i_itr}
            for k, v in out.items():
                if k in bar_keys:
                    bar[f'{self.name}_{k}'] = v
                else:
                    info[f'{self.name}_{k}'] = v
            self.bar_and_hist(bar)
            self.add_hist(info)
            self._flush()
