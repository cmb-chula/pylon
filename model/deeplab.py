import segmentation_models_pytorch as smp
from trainer.start import *

from .common import *


@dataclass
class Deeplabv3Config(BaseConfig):
    n_out: int
    backbone: str = 'resnet50'
    n_in: int = 1
    n_dec_ch: int = 256
    dilate: bool = False
    weights: str = 'imagenet'

    @property
    def name(self):
        name = f'deeplabv3+-{self.backbone}-dec{self.n_dec_ch}'
        if self.weights is not None:
            name += f'-{self.weights}'
        return name

    def make_model(self):
        return Deeplabv3(self)


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

    def forward(self, img, classification=None, **kwargs):
        # enforce float32 is a good idea
        # because if the loss function involves a reduction operation
        # it would be harmful, this prevents the problem
        seg = self.net(img).float()
        pred = self.pool(seg)
        pred = torch.flatten(pred, start_dim=1)

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
