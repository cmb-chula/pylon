from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from .callbacks.metric_cb import CollectCb
from .looper import *


class BasePredictor(LooperInterface):
    """
    loop over a dataloader in eval mode without gradients

    Args:
        collect_keys: keys to be collected (into a big tensor) from each forward, default "pred" 
    """
    def __init__(self,
                 trainer: BaseTrainer,
                 callbacks=[],
                 collect_keys=['pred']):
        self.trainer = trainer
        if len(collect_keys) > 0:
            callbacks += [CollectCb(collect_keys)]
        self.callbacks = callbacks
        self.looper = Looper(self,
                             net=self.trainer.net,
                             mode='eval',
                             callbacks=self.callbacks)

    def on_train_begin(self, **kwargs):
        # maintain the default behavior from the trainer
        return self.trainer.on_train_begin(**kwargs)

    def on_ep_begin(self, **kwargs):
        # maintain the default behavior from the trainer
        return self.trainer.on_ep_begin(**kwargs)

    @torch.no_grad()
    def forward_pass(self, data, **kwargs):
        # maintain the default behavior from the tranier
        return self.trainer.forward_pass(data, **kwargs)

    def backward_pass(self, forward, **kwargs):
        pass

    def optimize(self, **kwargs):
        pass

    def predict(self, loader: DataLoader):
        """
        Return:
            looper's buffer; default would contain "pred" key (from CollectorCb)
        """
        self.looper.loop(loader, n_max_itr=len(loader))
        return self.looper.buffer
