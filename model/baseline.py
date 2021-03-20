from trainer.start import *
from segmentation_models_pytorch.encoders import get_encoder
from utils.pretrain import *


@dataclass
class BaselineModelConfig(BaseConfig):
    n_out: int
    backbone: str = 'resnet50'
    n_in: int = 1
    weights: str = 'imagenet'
    pooling: str = 'maxpool'
    pretrain_conf: PretrainConfig() = None

    @property
    def name(self):
        name = f'baseline-{self.backbone}-{self.pooling}'
        if self.weights is not None:
            name += f'-{self.weights}'
        if self.pretrain_conf is not None:
            name += f'-{self.pretrain_conf.name}'
        return name


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

    def forward(self, x):
        # select the last layer
        x = self.net(x)[-1]
        seg = self.out(x).float()
        pred = self.pool(seg)
        pred = torch.flatten(pred, 1)
        return {
            'pred': pred,
            'seg': seg,
        }
