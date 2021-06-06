from .trainer_base import *
from .callbacks.base_cb import *
from .callbacks.common_cb import *

try:
    from torch.cuda.amp import GradScaler, autocast
except Exception:
    pass
from .callbacks.profiler_cb import *

__all__ = ['amp_trainer_mask']


def amp_trainer_mask(cls):
    class AMPTrainer(cls):
        """Pytorch's automatic mixed precision
        requires: pytorch >= 1.6
        see: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            print('init pytorch\'s amp')
            self.scaler = GradScaler()

        def forward_pass(self, data, **kwargs):
            with autocast():
                return super().forward_pass(data=data, **kwargs)

        def backward_pass(self, forward, **kwargs):
            loss = forward['loss']
            if loss is not None:
                assert loss.dim() == 0, "loss must be reduced"
                with time_elapsed_to_profiler('backward'):
                    self.opt.zero_grad()
                    self.scaler.scale(loss).backward()
                    # allow modifying the gradient directly
                    self.scaler.unscale_(self.opt)

        def optimize(self, forward, **kwargs):
            loss = forward['loss']
            if loss is not None:
                with time_elapsed_to_profiler('optimize'):
                    self.scaler.step(self.opt)
                    self.scaler.update()

        def get_state(self):
            # including the amp state
            state = super().get_state()
            state['scaler'] = self.scaler.state_dict()
            return state

        def load_state(self, state):
            # including the amp state
            super().load_state(state)
            print('loading pytorch\'s amp state ...')
            if 'scaler' in state:
                self.scaler.load_state_dict(state['scaler'])
            else:
                print('warning: scaler state is not available')

        def __repr__(self):
            return f'<AMPTrainer {super().__repr__()}>'

    AMPTrainer.__name__ = cls.__name__ + f'_amp'
    return AMPTrainer
