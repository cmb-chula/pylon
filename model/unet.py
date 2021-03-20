from trainer.start import *
import segmentation_models_pytorch as smp


@dataclass
class UnetConfig(BaseConfig):
    n_out: int
    backbone: str = 'resnet50'
    n_in: int = 1
    n_dec_ch: Tuple[int] = (256, 128, 64, 32, 16)
    weights: str = 'imagenet'
    pretrain_name: str = None
    pretrain_prefix: str = None

    @property
    def name(self):
        name = f'unet-{self.backbone}-('
        name += ','.join(str(x) for x in self.n_dec_ch)
        name += f')'
        if self.weights is not None:
            name += f'-{self.weights}'
        if self.pretrain_name is not None:
            name += f'-w{self.pretrain_name}'
        return name


class Unet(nn.Module):
    def __init__(self, conf: UnetConfig):
        super().__init__()
        self.net = smp.Unet(
            conf.backbone,
            in_channels=conf.n_in,
            encoder_weights=conf.weights,
            decoder_channels=conf.n_dec_ch,
            classes=conf.n_out,
        )
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        seg = self.net(x).float()
        x = self.pool(seg)
        x = torch.flatten(x, 1)
        return {
            'pred': x,
            'seg': seg,
        }
