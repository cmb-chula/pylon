from .trainer_base import *
from .callbacks.metric_cb import *
from .looper import *
from .predictor_base import BasePredictor


class ValidatePredictor(BasePredictor):
    """
    a predictor for validation process

    Note: validate predictor doesn't have collect keys
    """
    def __init__(self, trainer: BaseTrainer, callbacks=[]):
        super(BasePredictor, self).__init__()
        self.trainer = trainer
        self.callbacks = callbacks
        self.looper = Looper(base=self, callbacks=self.callbacks)

    def predict(self, loader: DataLoader):
        """validate predictor returns all the stats instead of the prediction"""
        self.looper.loop(loader, n_max_itr=len(loader))
        stats = StatsCallback.collect_latest(self.callbacks)

        # the keys in the callbacks should be visible on the progress bar
        # everything else is kept in the history (not visible)
        bar_keys = set()
        for cb in self.callbacks:
            if isinstance(cb, StatsCallback):
                bar_keys |= set(cb.stats.keys())
        bar_keys -= set(['i_itr'])

        # to appear on the progress bar
        bar_stats = {k: v for k, v in stats.items() if k in bar_keys}
        # to appear only in the history
        hist_stats = {k: v for k, v in stats.items() if k not in bar_keys}
        return bar_stats, hist_stats
