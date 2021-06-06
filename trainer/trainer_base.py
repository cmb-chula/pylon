import torch.nn.functional as F

from .callbacks.base_cb import *
from .callbacks.profiler_cb import *
from .callbacks.report_cb import *
from .config_base import *
from .looper import *

__all__ = ['BaseTrainer', 'BaseTrainerConfig']


@dataclass
class BaseTrainerConfig(BaseConfig):
    device: str


class BaseTrainer(LooperInterface):
    """
    Args:
        callbacks: list of callbacks, leave as None for defaults
    """
    def __init__(
            self,
            conf: BaseTrainerConfig,
    ):
        super().__init__()
        self.conf = conf
        self.net = None
        self.opt = None
        self.init_net_and_opt()

    def init_net_and_opt(self):
        # init net and opt
        # could be reran for resetting
        self.net = self.make_net().to(self.device)
        self.opt = self.make_opt(self.net)

    @property
    def i_itr(self):
        return self.state['i_itr']

    @property
    def device(self):
        return self.conf.device

    def make_default_callbacks(self):
        return [
            ProgressCb(),
            ReportItrCb(),
            ReportLoaderStatsCb(),
            BatchPerSecondCb(),
            ReportLRCb(),
        ]

    def make_net(self):
        """the default net function"""
        return None

    def make_opt(self, net: nn.Module):
        """the default opt function"""
        return None

    def on_train_begin(self, **kwargs):
        """this is useful to implement automatic mixed-precision"""
        pass

    def on_ep_begin(self, **kwargs):
        """if the forward_pass method alone is not enough, which is the case of language model"""
        pass

    def forward_pass(self, data, **kwargs) -> Dict:
        """forward pass of the model, must return a dictionary."""
        x, y = data
        with time_elapsed_to_profiler('forward'):
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
        loss = forward['loss']
        if loss is not None:
            assert forward['loss'].dim() == 0, "loss must be reduced"
            with time_elapsed_to_profiler('backward'):
                self.opt.zero_grad()
                forward['loss'].backward()

    def optimize(self, forward, **kwargs):
        loss = forward['loss']
        if loss is not None:
            with time_elapsed_to_profiler('optimize'):
                self.opt.step()

    def on_abrupt_end(self, **kwargs):
        # this rolls back the i_itr on the failed iteration
        self.state['i_itr'] = max(self.state['i_itr'] - 1, 0)

    def train(
            self,
            loader: DataLoader,
            n_max_itr: int = None,
            n_max_ep: int = None,
            callbacks=None,
    ) -> pd.DataFrame:
        """
        Args: 
            loader: data loader
            n_max_itr: number of iterations before terminate, None means using n_max_ep instead
            n_max_ep: number of epochs before terminate, None means using n_max_itr instead
            callbacks: 
        """
        if n_max_itr is None:
            assert n_max_ep is not None, 'either n_max_itr or n_max_ep must be present'
            n_max_itr = len(loader) * n_max_ep
        else:
            assert n_max_ep is None, 'the supplied n_max_ep has no effect'

        if callbacks is None:
            callbacks = self.make_default_callbacks()

        # looper is the one who calls all the methods in this trainer
        # looper supplies argument (also kwargs) into each method
        looper = Looper(self, callbacks=callbacks)

        # loop
        looper.loop(loader=loader, n_max_itr=n_max_itr)
        # collect the stats
        df = StatsCallback.combine_callbacks(callbacks)
        return df

    def get_state(self):
        return {
            'net': self.net.state_dict() if self.net is not None else None,
            'opt': self.opt.state_dict() if self.opt is not None else None,
            'state': self.state,
        }

    def load_state(self, state):
        # load the network state
        if self.net is not None:
            self.net.load_state_dict(state['net'])
        # load the optimizer state
        if self.opt is not None:
            if 'opt' not in state or state['opt'] is None:
                print('warning: cannot load the optimizer state!')
            else:
                self.opt.load_state_dict(state['opt'])
        # load the trainer state
        self.state.update(state['state'])

    def save(self, dirname: str):
        """save the trainer into trainer.pkl and model.pkl,
        model.pkl only contains the net."""
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        state = self.get_state()
        net = state['net']
        notnet = {k: v for k, v in state.items() if k != 'net'}
        self.conf.save(f'{dirname}/config.json')
        safe_torch_save(net, f'{dirname}/model.pkl')
        safe_torch_save(notnet, f'{dirname}/trainer.pkl')

    def load(self,
             dirname: str,
             load_config: bool = False,
             config_overload: dict = None):
        """load the trainer's state including net, opt, state"""
        if load_config:
            self.conf.load(f'{dirname}/config.json')
            if config_overload is not None:
                self.conf.__dict__.update(config_overload)
            # need to reinit the models and optimizers
            # due to possible config changes
            self.init_net_and_opt()

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
        """legacy: separate trainer file and model file"""
        assert '.pkl' in file
        dirname = os.path.dirname(file)
        state = torch.load(file, map_location='cpu')
        if 'net' not in state:
            print('already converted')
            return
        net = state['net']
        notnet = {k: v for k, v in state.items() if k != 'net'}
        safe_torch_save(net, f'{dirname}/model.pkl')
        safe_torch_save(notnet, f'{dirname}/trainer.pkl')

    def __repr__(self):
        return f'<BaseTrainer {self.state}>'

    def __str__(self):
        return self.__repr__()
