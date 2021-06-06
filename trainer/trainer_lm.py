import torch.nn.functional as F

from .trainer_base import *
from .callbacks.start import *

class LMTrainer(BaseTrainer):
    """language modeling trainer
    """
    def on_ep_begin(self, **kwargs):
        self.hidden = None

    def forward_pass(self, data):
        x, y = data
        pred, hidden = self.net(x, self.hidden)
        self.hidden = detach(hidden)
        # flatten pred and target before loss
        t, b, n_token = pred.shape
        loss = F.cross_entropy(pred.view(-1, n_token), y.view(-1))
        with torch.no_grad():
            ppl = torch.exp(loss)
        return {
            'x': x,
            'y': y,
            'pred': pred,
            'loss': loss,
            'ppl': ppl,
            'hidden': self.hidden,
            'n': y.shape[0] * y.shape[1]
        }

    def make_default_callbacks(self):
        return super().make_default_callbacks() + [MovingAvgCb(['loss', 'ppl'])]

    def __repr__(self):
        return f'<LMTrainer {self.looper.state}>'
