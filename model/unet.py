import segmentation_models_pytorch as smp
from trainer.start import *

from .common import *


@dataclass
class UnetConfig(BaseConfig):
    n_out: int
    backbone: str = 'resnet50'
    n_in: int = 1
    n_dec_ch: Tuple[int] = (256, 128, 64, 32, 16)
    weights: str = 'imagenet'

    @property
    def name(self):
        name = f'unet-{self.backbone}-('
        name += ','.join(str(x) for x in self.n_dec_ch)
        name += f')'
        if self.weights is not None:
            name += f'-{self.weights}'
        return name

    def make_model(self):
        return Unet(self)


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
