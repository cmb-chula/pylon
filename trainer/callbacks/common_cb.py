import torch.optim as optim

from .base_cb import *


class LRReducePlateauCb(Callback):
    """
    Args:
        key: the key to watch
        n_cycle: how frequent to check (need to match the validator), default: n_ep_itr
        **kwargs: see ReduceLROnPlateau on Pytorch
    """
    def __init__(self,
                 key,
                 n_cycle=None,
                 mode='max',
                 patience=10,
                 factor=0.2,
                 **kwargs):
        super().__init__()
        self.scheduler = None
        self.key = key
        self.n_cycle = n_cycle
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.kwargs = kwargs

    def get_state(self):
        assert self.scheduler is not None
        return {
            'self': super().get_state(),
            'scheduler': self.scheduler.state_dict(),
        }

    def load_state(self, state):
        assert self.scheduler is not None
        super().load_state(state['self'])
        self.scheduler.load_state_dict(state['scheduler'])

    # should load before resume
    @set_order(0)
    def on_train_begin(self, trainer, n_ep_itr, **kwargs):
        if self.n_cycle is None:
            # set default to 1 epoch
            self.n_cycle = n_ep_itr

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            trainer.opt,
            mode=self.mode,
            patience=self.patience,
            factor=self.factor,
            **self.kwargs,
        )

    def on_batch_end(self, callbacks, i_itr, **kwargs):
        if i_itr % self.n_cycle == 0:
            v = None
            for cb in callbacks:
                if isinstance(cb, StatsCallback):
                    if self.key in cb.stats:
                        v = cb.stats[self.key]
                        break
            assert v is not None, "needs to set the cycle to match the validator callback"
            self.scheduler.step(v)


class TerminateLRCb(Callback):
    def __init__(self, lr_thresh, begin=0):
        super().__init__()
        self.lr_thresh = lr_thresh
        self.begin = begin

    def on_batch_end(self, trainer, i_itr, **kwargs):
        if i_itr >= self.begin:
            lr = trainer.opt.param_groups[0]['lr']
            if lr <= self.lr_thresh:
                print(f'terminated because lr {lr} <= {self.lr_thresh}')
                raise KeyboardInterrupt()
