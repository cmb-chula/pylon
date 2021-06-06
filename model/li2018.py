from trainer.start import *
from segmentation_models_pytorch.encoders import get_encoder

from .common import *


@dataclass
class Li2018Config(BaseConfig):
    n_out: int
    backbone: str = 'resnet50'
    n_in: int = 1
    n_dec_ch: int = 512
    out_size: int = 20
    pooling: str = 'milpool'
    min_val: float = 0.98
    weights: str = 'imagenet'

    @property
    def name(self):
        name = f'li2018-{self.backbone}-dec{self.n_dec_ch}out{self.out_size}'
        if self.pooling == 'milpool':
            name += f'-mil{self.min_val}'
        else:
            name += f'-{self.pooling}'
        if self.weights is not None:
            name += f'-{self.weights}'
        return name

    def make_model(self):
        return Li2018(self)


class Li2018(nn.Module):
    def __init__(self, conf: Li2018Config):
        super().__init__()
        self.net = get_encoder(
            name=conf.backbone,
            in_channels=conf.n_in,
            weights=conf.weights,
        )

        self.out = nn.Sequential(
            nn.UpsamplingBilinear2d((conf.out_size, conf.out_size)),
            nn.Conv2d(self.net.out_channels[-1], conf.n_dec_ch, 3, padding=1),
            nn.BatchNorm2d(conf.n_dec_ch),
            nn.ReLU(),
            nn.Conv2d(conf.n_dec_ch, conf.n_out, 1, bias=True),
        )

        pooling_opts = {
            'maxpool':
            nn.AdaptiveMaxPool2d(1),
            'milpool':
            MILPool(min_val=conf.min_val, apply_sigmoid=True, ret_logit=True),
        }
        self.pool = pooling_opts[conf.pooling]

    def forward(self, img, classification=None, **kwargs):
        # enforce float32 is a good idea
        # because if the loss function involves a reduction operation
        # it would be harmful, this prevents the problem
        seg = self.net(img)[-1]
        seg = self.out(seg).float()
        pred = self.pool(seg)
        pred = torch.flatten(pred, start_dim=1)

        loss = None
        loss_pred = None
        if classification is not None:
            loss_pred = F.binary_cross_entropy_with_logits(
                pred, classification.float())
            loss = loss_pred

        return ModelReturn(
            pred=pred,
            pred_seg=seg,
            loss=loss,
            loss_pred=loss_pred,
            loss_bbox=None,
        )


def mil_output(p, min_val):
    """
    Args:
        min_val: cap the min value of 1-p to prevent underflow
    """
    n, c, _, _ = p.shape
    not_p = 1 - p
    not_p = (1 - min_val) * not_p + min_val
    not_p = not_p.view(n, c, -1).float()
    pred = 1 - torch.prod(not_p, dim=-1, keepdim=True)
    pred = pred.view(n, c, 1, 1)
    return pred


class MILPool(nn.Module):
    """
    Multi-instance pooling:
    The output is positive when there is at least one positive patch
    
    Found in:
    Li, Zhe, Chong Wang, Mei Han, Yuan Xue, Wei Wei, Li-Jia Li, and Li Fei-Fei. 2018. 
    “Thoracic Disease Identification and Localization with Limited Supervision.” 
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 8290–99.

    Args:
        min_val: None = auto
        ret_logit: returns as logit (not prob), to keep the interface invariance
    """
    def __init__(self, min_val=0.98, apply_sigmoid=True, ret_logit=False):
        super().__init__()
        self.min_val = min_val
        self.apply_sigmoid = apply_sigmoid
        self.ret_logit = ret_logit

    def forward(self, x):
        n, c, h, w = x.shape
        if self.apply_sigmoid:
            x = torch.sigmoid(x)
        pred = mil_output(x, min_val=self.min_val)
        if self.ret_logit:
            # logit function inverses the sigmoid
            pred = torch.log(pred / (1 - pred))
        return pred