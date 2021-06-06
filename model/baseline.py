from segmentation_models_pytorch.encoders import get_encoder
from trainer.start import *
from utils.pretrain import *

from .common import *


@dataclass
class BaselineModelConfig(BaseConfig):
    n_out: int
    backbone: str = 'resnet50'
    n_in: int = 1
    weights: str = 'imagenet'
    pooling: str = 'maxpool'
    pretrain_conf: PretrainConfig = None

    @property
    def name(self):
        name = f'baseline-{self.backbone}-{self.pooling}'
        if self.weights is not None:
            name += f'-{self.weights}'
        if self.pretrain_conf is not None:
            name += f'-{self.pretrain_conf.name}'
        return name

    def make_model(self):
        return BaselineModel(self)


class BaselineModel(nn.Module):
    def __init__(self, conf: BaselineModelConfig):
        super().__init__()
        self.conf = conf
        self.net = get_encoder(
            name=conf.backbone,
            in_channels=conf.n_in,
            weights=conf.weights,
        )

        self.out = nn.Conv2d(self.net.out_channels[-1],
                             conf.n_out,
                             kernel_size=1,
                             bias=True)

        pooling_opts = {
            'maxpool': nn.AdaptiveMaxPool2d(1),
            'avgpool': nn.AdaptiveAvgPool2d(1),
        }
        self.pool = pooling_opts[conf.pooling]

        if conf.pretrain_conf is not None:
            load_pretrain(conf.pretrain_conf, target=self)

    def forward(self, img, classification=None, **kwargs):
        # select the last layer
        feat = self.net(img)[-1]
        if self.conf.pooling == 'maxpool':
            # (bs, cls, h, w)
            seg = self.out(feat).float()
            # (bs, cls, 1, 1)
            pred = self.pool(seg)
            # (bs, cls)
            pred = torch.flatten(pred, 1)
        elif self.conf.pooling == 'avgpool':
            # (bs, c, 1, 1)
            pred = self.pool(feat)
            # (bs, cls, 1, 1)
            pred = self.out(pred).float()
            # (bs, cls)
            pred = torch.flatten(pred, 1)
            # (bs, cls, h, w)
            seg = self.out(feat).float()
        else:
            raise NotImplementedError()

        loss = None
        loss_pred = None
        loss_bbox = None
        if classification is not None:
            loss_pred = F.binary_cross_entropy_with_logits(
                pred, classification.float())
            loss = loss_pred

        return ModelReturn(
            pred=pred,
            pred_seg=seg,
            loss=loss,
            loss_pred=loss_pred,
            loss_bbox=loss_bbox,
        )
