import segmentation_models_pytorch as smp
from trainer.start import *


@dataclass
class Deeplabv3Config(BaseConfig):
    n_out: int
    backbone: str = 'resnet50'
    n_in: int = 1
    n_dec_ch: int = 256
    dilate: bool = False
    weights: str = 'imagenet'
    pretrain_name: str = None
    pretrain_prefix: str = None

    @property
    def name(self):
        name = f'deeplabv3+-{self.backbone}-dec{self.n_dec_ch}'
        if self.weights is not None:
            name += f'-{self.weights}'
        if self.pretrain_name is not None:
            name += f'-w{self.pretrain_name}'
        return name


class Deeplabv3(nn.Module):
    def __init__(self, conf: Deeplabv3Config):
        super().__init__()
        self.net = smp.DeepLabV3Plus(conf.backbone,
                                     encoder_weights=conf.weights,
                                     in_channels=conf.n_in,
                                     decoder_channels=conf.n_dec_ch,
                                     classes=conf.n_out,
                                     upsampling=1)
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        seg = self.net(x).float()
        x = self.pool(seg)
        x = torch.flatten(x, 1)
        return {
            'pred': x,
            'seg': seg,
        }
