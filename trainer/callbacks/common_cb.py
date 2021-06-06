from torch import optim

from ..params_grads import *
from .base_cb import *


class GradientClipCb(Callback):
    """Supports Apex's AMP"""
    def __init__(self, clip_norm, **kwargs):
        super().__init__(**kwargs)
        self.clip_norm = clip_norm

    def on_backward_end(self, trainer, **kwargs):
        nn.utils.clip_grad_norm_(iter_opt_params(trainer.opt),
                                 max_norm=self.clip_norm)


class LRSchedulerCb(Callback):
    """
    Args:
        lr_fn: learning rate function (p, i, i_ep, loss) -> (float, None); None to ignore.
    """
    def __init__(self,
                 lr_fn,
                 n_cycle_itr: int = 1,
                 n_cycle_ep: float = None,
                 loss_key='loss',
                 **kwargs):
        super().__init__(**kwargs)
        self.lr_fn = lr_fn
        assert n_cycle_itr is not None or n_cycle_ep is not None, f'need to specify the cycle'
        self.n_cycle_itr = n_cycle_itr
        self.n_cycle_ep = n_cycle_ep
        self.loss_key = loss_key

    def on_train_begin(self, n_ep_itr, **kwargs):
        super().on_train_begin(n_ep_itr=n_ep_itr, **kwargs)
        if self.n_cycle_itr is None:
            self.n_cycle_itr = int(self.n_cycle_ep * n_ep_itr)

    def on_step_begin(self, trainer, n_max_itr, i_itr, n_ep_itr, callbacks,
                      **kwargs):
        if i_itr % self.n_cycle_itr == 0:
            try:
                loss = get_val_from_statcbs(self.loss_key, callbacks)
            except ValueError:
                loss = None
            # f_ep should start from 0.00 and is float
            f_ep = i_itr / n_ep_itr
            n_max_ep = n_max_itr / n_ep_itr
            scale = self.lr_fn(
                p=i_itr / n_max_itr,
                i=i_itr,
                n_max_itr=n_max_itr,
                f_ep=f_ep,
                n_max_ep=n_max_ep,
                loss=loss,
            )
            # only care the float scales
            if isinstance(scale, (int, float)):
                for g in trainer.opt.param_groups:
                    assert 'lr' in g, "the optimizer doesn't seed to have the lr option"
                    if 'base_lr' not in g:
                        g['base_lr'] = g['lr']
                    g['lr'] = g['base_lr'] * scale


class CosineAnnealingWarmRestartsLRCb(Callback):
    def __init__(self, T_0, T_mul=1, eta_min=0, last_epoch=-1, verbose=False):
        super().__init__()
        self.scheduler = None
        self.T_0 = T_0
        self.T_mul = T_mul
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.verbose = verbose

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
    def on_train_begin(self, trainer, **kwargs):
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            trainer.opt,
            T_0=self.T_0,
            T_mult=self.T_mul,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch,
            verbose=self.verbose,
        )

    def on_batch_end(self, f_ep, **kwargs):
        self.scheduler.step(f_ep)


class LRReducePlateauCb(Callback):
    """
    Args:
        key: the key to watch
        mode: 'max' or 'min'
        n_cycle: how frequent to check (need to match the validator), default: n_ep_itr
        patience: how many cycles to wait for an improvement
        **kwargs: see ReduceLROnPlateau on Pytorch
    """
    def __init__(self,
                 key,
                 mode,
                 n_itr_cycle: int = None,
                 n_ep_cycle: float = None,
                 patience: int = 10,
                 factor=0.2,
                 n_start_ep: int = 0,
                 **kwargs):
        super().__init__()
        self.scheduler = None
        self.key = key
        self.n_itr_cycle = n_itr_cycle
        self.n_ep_cycle = n_ep_cycle
        self.n_start_ep = n_start_ep
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
        if self.n_itr_cycle is None:
            if self.n_ep_cycle is not None:
                self.n_itr_cycle = int(self.n_ep_cycle * n_ep_itr)
            else:
                # default to 1 ep
                self.n_itr_cycle = n_ep_itr

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            trainer.opt,
            mode=self.mode,
            patience=self.patience,
            factor=self.factor,
            **self.kwargs,
        )

    def on_batch_end(self, callbacks, i_itr, i_ep, **kwargs):
        if i_itr % self.n_itr_cycle == 0 and i_ep >= self.n_start_ep:
            # getting the key value from the stats
            v = get_val_from_statcbs(self.key, callbacks)
            self.scheduler.step(v)


class WeightPolyakCb(Callback):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.w = None
        self.rate = rate

    def get_params(self, net):
        return nn.utils.parameters_to_vector(net.parameters())

    def on_step_begin(self, trainer, **kwargs):
        # get params
        self.w = self.get_params(trainer.net)

    @torch.no_grad()
    def on_step_end(self, trainer, **kwargs):
        # update w
        new_w = self.get_params(trainer.net)
        new_w = self.rate * self.w + (1 - self.rate) * new_w
        nn.utils.vector_to_parameters(new_w, trainer.net.parameters())
        self.w = None


class GracefulException(Exception):
    pass


class TerminateLRException(GracefulException):
    pass


class TerminateLRCb(Callback):
    """
    Args:
        lr_thresh: <= which will terminate
        begin_itr: not terminate until the specified itr
    """
    def __init__(self, lr_thresh: float, begin_itr: int = 0):
        super().__init__()
        self.lr_thresh = lr_thresh
        self.begin_itr = begin_itr

    def on_batch_end(self, trainer, i_itr, **kwargs):
        if i_itr >= self.begin_itr:
            lr = trainer.opt.param_groups[0]['lr']
            if lr <= self.lr_thresh:
                raise TerminateLRException(
                    f'terminated because lr {lr} <= {self.lr_thresh}')


class EarlyStopException(GracefulException):
    pass


class EarlyStopCb(Callback):
    def __init__(self,
                 patience: int,
                 n_ep_cycle=1,
                 metric='val_loss',
                 metric_mode='min',
                 verbose=False):
        super().__init__()
        assert metric_mode in ('max', 'min')
        self.patience = patience
        self.metric = metric
        self.metric_mode = metric_mode
        self.verbose = verbose
        self.n_ep_cycle = n_ep_cycle

        self._state['best'] = None
        self._state['best_i'] = None
        self._state['i'] = 0

    def on_train_begin(self, n_ep_itr, **kwargs):
        self.n_itr_cycle = self.n_ep_cycle * n_ep_itr

    def on_batch_end(self, trainer, i_itr, n_ep_itr, callbacks, **kwargs):
        if i_itr % self.n_itr_cycle == 0:
            self.i += 1
            if self.verbose:
                print(f'early stop: round {self.i}')
            v = get_val_from_statcbs(self.metric, callbacks)
            if self.best is None:
                self.best = v
                self.best_i = self.i
            else:
                if self.metric_mode == 'max':
                    if v > self.best:
                        self.best = v
                        self.best_i = self.i
                elif self.metric_mode == 'min':
                    if v < self.best:
                        self.best = v
                        self.best_i = self.i
                else:
                    raise NotImplementedError()

            if self.verbose:
                print(
                    f'early stop: metric {v}, best {self.best} (round {self.best_i})'
                )
                print(f'early stop: gap {self.i - self.best_i}')
            if self.i - self.best_i > self.patience:
                raise EarlyStopException('early stop')


class StopAnyTimeCb(Callback):
    """supress the keyboard interrupt allowing to stop the training anytime
    while getting the return results"""
    def on_abrupt_end(self, e, **kwargs):
        if isinstance(e, KeyboardInterrupt):
            # suppress the raise
            return True
        else:
            # cannot suppress errors
            return False


class AutoInterruptException(GracefulException):
    pass


class AutoInterrupt(Callback):
    """raises a KeyboardInterrupt at n_itr, useful for playing around."""
    def __init__(self, n_itr, order=None):
        super().__init__(order=order)
        self.n_itr = n_itr

    def on_batch_begin(self, i_itr, **kwargs):
        # this will allow for the validate to end from the last itr
        if i_itr >= self.n_itr:
            raise AutoInterruptException(
                f'auto interrupt at i_itr = {self.n_itr}')
