import torch.nn.functional as F
from .trainer_base import *
from .callbacks.metric_cb import *

class MultiClassTrainer(BaseTrainer):
    def forward_pass(self, data, **kwargs):
        x, y = data
        pred = self.net(x)
        loss = F.cross_entropy(pred, y)
        acc = accuracy(pred, y)
        return {
            'x': x,
            'y': y,
            'loss': loss,
            'acc': acc,
            'n': len(x),
        }

    @classmethod
    def make_default_callbacks(cls):
        default = super().make_default_callbacks()
        return default + [
            MovingAvgCb(['loss', 'acc']),
            BatchPerSecondCb(),
        ]

    def __repr__(self):
        return f'<MultiClassTrainer {self.looper.state}>'

def accuracy(pred, y, ignore_index=None):
    """
    Args:
        ignore_index: skip this y value
    """
    assert pred.dim() == 2
    assert y.dim() == 1
    if ignore_index is not None:
        idx = y != ignore_index
        if idx.sum() == 0:
            return torch.tensor(0.).to(pred.device)
        pred = pred[idx]
        y = y[idx]
    return (torch.argmax(pred, dim=1) == y).float().mean()
