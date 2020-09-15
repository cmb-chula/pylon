from typing import List

from torch.utils.data import DataLoader

from .callbacks.base_cb import *
from .set_mode import *


class LooperInterface:
    """a base for looper should have the following interface"""
    def on_train_begin(self, **kwargs):
        pass

    def on_ep_begin(self, **kwargs):
        pass

    def forward_pass(self, data, **kwargs):
        pass

    def backward_pass(self, forward, **kwargs):
        pass

    def optimize(self, **kwargs):
        pass


class Looper:
    """
    looper lopps over a loader with predefined number of iterations
    Goal: removing the duplicated parts between trainer and predictor

    Args:
        predictor: a class with "on_ep_begin", "forward_pass", "backward_pass", "optimize" methods
        callbacks: 
    """
    def __init__(
            self,
            base: LooperInterface,
            net: nn.Module,
            mode: str,
            callbacks: List[Callback],
    ):
        self.base = base
        self.net = net
        self.mode = mode
        self.callbacks = callbacks

        self.state = {'i_itr': 0}
        # buffer collects outputs from the model
        # this is useful for calculating dataset-wise metrics, like BLEU or AUROC
        # callbacks populate data in buffer
        self.buffer = defaultdict(list)

    def one_batch(self, data):
        # start of iteration
        self.state['i_itr'] += 1
        self('on_batch_begin', data=data)
        # forward pass
        self('on_forward_begin', data=data)
        with set_mode(self.net, self.mode):
            forward = self.base.forward_pass(data, **self._kwargs())
        self('on_forward_end', data=data, forward=forward)
        # backward pass
        self('on_backward_begin', data=data, forward=forward)
        self.base.backward_pass(forward, **self._kwargs())
        self('on_backward_end', data=data, forward=forward)
        # step the optimizer
        self('on_step_begin', data=data, forward=forward)
        self.base.optimize(**self._kwargs())
        self('on_step_end', data=data, forward=forward)
        self('on_batch_end', data=data, forward=forward)

        # iteration exceeds
        if self.state['i_itr'] >= self.n_max_itr:
            raise StopIteration()

    def one_epoch(self, loader):
        self.base.on_ep_begin(**self._kwargs())
        self('on_ep_begin')
        try:
            for data in loader:
                self.one_batch(data)
        finally:
            self('on_ep_end')

    def loop(self, loader: DataLoader, n_max_itr: int):
        self.loader = loader
        self.n_max_itr = n_max_itr

        try:
            self.base.on_train_begin(**self._kwargs())
            self('on_train_begin')
            while self.state['i_itr'] < self.n_max_itr:
                self.one_epoch(self.loader)
            self('on_train_end')
        except StopIteration:
            # normal stopping
            self('on_train_end')
        except KeyboardInterrupt as e:
            # run the train_end before exiting (saving etc.)
            print('keyboard interrupt - wait for the finalizations')
            # this allows the callback to suppress the keyboard interrupt
            if not self('on_train_end', e=e):
                raise
        except Exception as e:
            # in other cases, we should not call the train_end to slow things down
            print('unexpected exception:', e)
            self('on_abrupt_end', e=e)
            raise e

    def _kwargs(self, kwargs=dict()):
        """these will be supplied to callbacks,
        these are variables that are expected to be used by any callback"""
        n_ep_itr = len(self.loader)
        kwargs = {
            'trainer': self.base,
            'looper': self,
            'loader': self.loader,
            'n_max_itr': self.n_max_itr,
            'n_ep_itr': n_ep_itr,
            'callbacks': self.callbacks,
            'i_ep': int(self.state['i_itr'] / n_ep_itr) + 1,
            'p_ep': (self.state['i_itr'] % n_ep_itr) / n_ep_itr * 100,
            'buffer': self.buffer,
            **self.state,
            **kwargs,
        }
        return kwargs

    def __call__(self, event, **kwargs):
        """call event callbacks"""
        return callback_call(callbacks=self.callbacks,
                             method=event,
                             kwargs=self._kwargs(kwargs))
