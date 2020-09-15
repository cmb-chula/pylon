from datetime import datetime
from typing import Callable

import torch.nn.functional as F

from .callbacks.base_cb import *
from .callbacks.report_cb import *
from .looper import *


class BaseTrainer(LooperInterface):
    """
    Args:
        callbacks: list of callbacks, leave as None for defaults
    """
    def __init__(
            self,
            net_fn: Callable,
            opt_fn: Callable,
            device: str,
            callbacks=None,
    ):
        self.net_fn = net_fn
        self.opt_fn = opt_fn
        self.device = device

        if net_fn is not None:
            self.net = net_fn().to(self.device)
        else:
            self.net = None

        if opt_fn is not None:
            self.opt = opt_fn(self.net)
        else:
            self.opt = None

        # after everything is ready
        if callbacks is None:
            callbacks = self.make_default_callbacks()
        self.callbacks = callbacks

        # looper
        self.looper = Looper(self,
                             net=self.net,
                             mode='train',
                             callbacks=callbacks)

    @property
    def i_itr(self):
        return self.looper.state['i_itr']

    @property
    def state(self):
        return self.looper.state

    @classmethod
    def make_default_callbacks(cls):
        return [
            ProgressCb(),
            ReportItrCb(),
        ]

    def on_train_begin(self, **kwargs):
        """this is useful to implement apex"""
        pass

    def on_ep_begin(self, **kwargs):
        """if the forward_pass method alone is not enough, which is the case of language model"""
        pass

    def forward_pass(self, data, **kwargs):
        x, y = data
        pred = self.net(x)
        loss = F.cross_entropy(pred, y)
        return {
            'x': x,
            'y': y,
            'pred': pred,
            'loss': loss,
            'n': len(y),
        }

    def backward_pass(self, forward, **kwargs):
        assert forward['loss'].dim() == 0, "loss must be reduced"
        self.opt.zero_grad()
        forward['loss'].backward()

    def optimize(self, **kwargs):
        self.opt.step()

    def train(
            self,
            loader: DataLoader,
            n_max_itr: int = None,
            n_max_ep: int = None,
    ) -> pd.DataFrame:
        """
        Args: 
            loader: data loader
            n_max_itr: number of iterations before terminate, None means using n_max_ep instead
            n_max_ep: number of epochs before terminate, None means using n_max_itr instead
        """
        if n_max_itr is None:
            assert n_max_ep is not None, 'either n_max_itr or n_max_ep must be present'
            n_max_itr = len(loader) * n_max_ep
        else:
            assert n_max_ep is None, 'the supplied n_max_ep has no effect'

        # loop
        self.looper.loop(loader=loader, n_max_itr=n_max_itr)
        # collect the stats
        df = StatsCallback.combine_callbacks(self.callbacks)
        return df

    def get_state(self):
        return {
            'net': self.net.state_dict() if self.net is not None else None,
            'opt': self.opt.state_dict() if self.opt is not None else None,
            'state': self.state,
        }

    def load_state(self, state):
        if self.net is not None:
            self.net.load_state_dict(state['net'])
        # only load if we have an optimizer
        if self.opt is not None:
            if 'opt' not in state or state['opt'] is None:
                print('warning: cannot load the optimizer state!')
            else:
                self.opt.load_state_dict(state['opt'])
        self.state.update(state['state'])

    def save(self, dirname: str):
        """save the trainer into trainer.pkl and model.pkl,
        model.pkl only contains the net."""
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        state = self.get_state()
        net = state['net']
        notnet = {k: v for k, v in state.items() if k != 'net'}
        torch_save(net, f'{dirname}/model.pkl')
        torch_save(notnet, f'{dirname}/trainer.pkl')

    def load(self, dirname: str):
        if dirname[-4:] == '.pkl':
            # this is old version, loads the whole state
            data = torch.load(dirname, map_location=self.device)
            self.load_state(data)
        else:
            # new version, load from two files
            state = {
                'net': torch.load(f'{dirname}/model.pkl',
                                  map_location=self.device),
                **torch.load(f'{dirname}/trainer.pkl',
                             map_location=self.device),
            }
            self.load_state(state)

    @classmethod
    def convert_save(cls, file: str):
        """separate trainer file and model file"""
        assert '.pkl' in file
        dirname = os.path.dirname(file)
        state = torch.load(file, map_location='cpu')
        if 'net' not in state:
            print('already converted')
            return
        net = state['net']
        notnet = {k: v for k, v in state.items() if k != 'net'}
        torch_save(net, f'{dirname}/model.pkl')
        torch_save(notnet, f'{dirname}/trainer.pkl')

    def __repr__(self):
        return f'<BaseTrainer {self.looper.state}>'

    def __str__(self):
        return self.__repr__()
