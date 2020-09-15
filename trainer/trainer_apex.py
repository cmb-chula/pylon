import torch
from apex import amp, optimizers


def apex_trainer_mask(cls, opt_level='O1'):
    class ApexTrainer(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.has_init_apex = False

        def on_train_begin(self, **kwargs):
            assert hasattr(self, 'opt'), "trainer must have an optimizer"
            # apex can init only once
            # validation will recall on_train_begin again
            # at which it should be suppressed
            if not self.has_init_apex:
                print('initializing amp ...')
                # set default cuda device (needed for apex)
                torch.cuda.set_device(self.device)
                self.net, self.opt = amp.initialize(self.net,
                                                    self.opt,
                                                    opt_level=opt_level)
                self.has_init_apex = True

        def backward_pass(self, forward, **kwargs):
            self.opt.zero_grad()
            loss = forward['loss']
            assert loss.dim() == 0, "loss must be reduced"
            with amp.scale_loss(loss, self.opt) as scaled_loss:
                scaled_loss.backward()

        def get_state(self):
            # including the amp state
            state = super().get_state()
            state['amp'] = amp.state_dict()
            return state

        def load_state(self, state):
            # including the amp state
            super().load_state(state)
            print('loading amp state ...')
            amp.load_state_dict(state['amp'])

        def __repr__(self):
            return f'<ApexTrainer {super().__repr__()} opt_level={opt_level}>'

    ApexTrainer.__name__ = cls.__name__ + f'_mixfp{opt_level}'
    return ApexTrainer
