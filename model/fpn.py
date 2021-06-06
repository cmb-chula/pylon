from segmentation_models_pytorch.base import (ClassificationHead,
                                              SegmentationHead,
                                              SegmentationModel)
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.fpn.decoder import *
from trainer.start import *

from .common import *


@dataclass
class FPNConfig(BaseConfig):
    n_out: int
    backbone: str = 'resnet50'
    n_in: int = 1
    n_pyramid_ch: int = 256
    n_dec_ch: int = 128
    weights: str = 'imagenet'
    # 'original', 'custom'
    segment_block: str = 'original'
    # 'groupnorm', 'batchnorm'
    use_norm: str = 'groupnorm'
    n_group: int = 32
    decoder_activation: str = 'relu'

    @property
    def name(self):
        name = f'fpn-{self.backbone}-py{self.n_pyramid_ch}dec{self.n_dec_ch}'
        if self.segment_block != 'original':
            if self.use_norm == 'groupnorm':
                name += f'-gn{self.n_group}'
            elif self.use_norm == 'batchnorm':
                name += f'-bn'
            else:
                raise NotImplementedError()
        if self.decoder_activation != 'relu':
            name += f'-{self.decoder_activation}'
        if self.weights is not None:
            name += f'-{self.weights}'
        return name

    def make_model(self):
        return FPN(self)


class FPN(nn.Module):
    def __init__(self, conf: FPNConfig):
        super().__init__()
        self.conf = conf
        self.net = FPNCustom(
            conf.backbone,
            in_channels=conf.n_in,
            encoder_weights=conf.weights,
            classes=conf.n_out,
            upsampling=1,
            decoder_dropout=0,
            decoder_pyramid_channels=conf.n_pyramid_ch,
            decoder_segmentation_channels=conf.n_dec_ch,
            segment_block=conf.segment_block,
            use_norm=conf.use_norm,
            n_group=conf.n_group,
            decoder_activation=conf.decoder_activation,
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


class FPNCustom(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        segment_block='original',
        use_norm='groupnorm',
        decoder_activation='relu',
        decoder_negative_slope=0.01,
        n_group=32,
        **kwargs,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            segment_block=segment_block,
            use_norm=use_norm,
            activation=decoder_activation,
            negative_slope=decoder_negative_slope,
            n_group=n_group,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "fpn-{}".format(encoder_name)
        self.initialize()


class FPNDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=128,
        segment_block='original',
        dropout=0.2,
        merge_policy="add",
        use_norm='groupnorm',
        activation='relu',
        negative_slope=0.01,
        n_group=32,
    ):
        super().__init__()

        self.out_channels = segmentation_channels
        if encoder_depth < 3:
            raise ValueError(
                "Encoder depth for FPN decoder cannot be less than 3, got {}.".
                format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0],
                            pyramid_channels,
                            kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        seg_opts = {
            'original': SegmentationBlock,
            'simple': SegmentationBlockSimple,
            'custom': SegmentatioBlockCustom,
        }

        if segment_block == 'custom':
            seg_args = dict(use_norm=use_norm,
                            activation=activation,
                            negative_slope=negative_slope,
                            n_group=n_group)
        else:
            seg_args = {}

        self.seg_blocks = nn.ModuleList([
            seg_opts[segment_block](
                pyramid_channels,
                segmentation_channels,
                n_upsamples=n_upsamples,
                **seg_args,
            ) for n_upsamples in [3, 2, 1, 0]
        ])

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [
            seg_block(p)
            for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])
        ]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x


class SegmentatioBlockCustom(nn.Module):
    """able to change norm"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_upsamples=0,
                 use_norm='groupnorm',
                 activation='relu',
                 negative_slope=0.01,
                 n_group=32):
        super().__init__()

        blocks = [
            Conv3x3NormReLU(in_channels,
                            out_channels,
                            upsample=bool(n_upsamples),
                            use_norm=use_norm,
                            activation=activation,
                            negative_slope=negative_slope,
                            n_group=n_group)
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(
                    Conv3x3NormReLU(out_channels,
                                    out_channels,
                                    upsample=True,
                                    use_norm=use_norm,
                                    activation=activation,
                                    negative_slope=negative_slope,
                                    n_group=n_group))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class SegmentationBlockSimple(nn.Module):
    """simple segmentation block has only one conv layer + upsample (the rest)"""
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        self.n_upsamples = n_upsamples
        self.block = Conv3x3GNReLU(in_channels, out_channels, upsample=False)

    def forward(self, x):
        x = self.block(x)
        for i in range(self.n_upsamples):
            x = F.interpolate(x,
                              scale_factor=2,
                              mode="bilinear",
                              align_corners=True)
        return x


class Conv3x3NormReLU(nn.Module):
    """custom the norm"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 upsample=False,
                 use_norm='groupnorm',
                 activation='relu',
                 negative_slope=0.01,
                 n_group=32):
        super().__init__()
        self.upsample = upsample

        norm_opts = {
            'groupnorm': lambda: nn.GroupNorm(n_group, out_channels),
            'batchnorm': lambda: nn.BatchNorm2d(out_channels),
        }

        act_opts = {
            'relu':
            lambda: nn.ReLU(inplace=True),
            'lrelu':
            lambda: nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        }

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels, (3, 3),
                      stride=1,
                      padding=1,
                      bias=False),
            norm_opts[use_norm](),
            act_opts[activation](),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x,
                              scale_factor=2,
                              mode="bilinear",
                              align_corners=True)
        return x
