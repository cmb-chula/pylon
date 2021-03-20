import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import (ClassificationHead,
                                              SegmentationHead,
                                              SegmentationModel)
from segmentation_models_pytorch.deeplabv3 import decoder
from segmentation_models_pytorch.encoders import get_encoder
from trainer.start import *


@dataclass
class Deeplabv3CustomConfig(BaseConfig):
    n_out: int
    backbone: str = 'resnet50'
    n_in: int = 1
    n_dec_ch: int = 256
    aspp_mode: str = 'original'
    dilate: bool = False
    weights: str = 'imagenet'
    pretrain_name: str = None
    pretrain_prefix: str = None

    @property
    def name(self):
        name = f'deeplabv3+-{self.backbone}-dec{self.n_dec_ch}'
        name += f'-{self.aspp_mode}'
        if self.weights is not None:
            name += f'-{self.weights}'
        if self.pretrain_name is not None:
            name += f'-w{self.pretrain_name}'
        return name


class Deeplabv3Custom(nn.Module):
    def __init__(self, conf: Deeplabv3CustomConfig):
        super().__init__()
        self.net = DeepLabV3PlusCustom(conf.backbone,
                                       encoder_weights=conf.weights,
                                       in_channels=conf.n_in,
                                       decoder_channels=conf.n_dec_ch,
                                       classes=conf.n_out,
                                       upsampling=1,
                                       aspp_mode=conf.aspp_mode)
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        seg = self.net(x).float()
        x = self.pool(seg)
        x = torch.flatten(x, 1)
        return {
            'pred': x,
            'seg': seg,
        }


class DeepLabV3PlusCustom(SegmentationModel):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            encoder_output_stride: int = 16,
            decoder_channels: int = 256,
            decoder_atrous_rates: tuple = (12, 24, 36),
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None,
            aspp_mode='original',
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        if encoder_output_stride == 8:
            self.encoder.make_dilated(stage_list=[4, 5], dilation_list=[2, 4])

        elif encoder_output_stride == 16:
            self.encoder.make_dilated(stage_list=[5], dilation_list=[2])
        else:
            raise ValueError(
                "Encoder output stride should be 8 or 16, got {}".format(
                    encoder_output_stride))

        self.decoder = DeepLabV3PlusDecoderCustom(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            aspp_mode=aspp_mode,
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


class DeepLabV3PlusDecoderCustom(nn.Module):
    def __init__(
            self,
            encoder_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
            aspp_mode='original',
    ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(
                output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride

        aspp_opts = {
            'original': decoder.ASPP,
            'nogap': ASPPNoGAP,
        }

        self.aspp = nn.Sequential(
            aspp_opts[aspp_mode](encoder_channels[-1],
                                 out_channels,
                                 atrous_rates,
                                 separable=True),
            decoder.SeparableConv2d(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels,
                      highres_out_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            decoder.SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features


class ASPPNoGAP(nn.Module):
    """no global pooling path"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 atrous_rates,
                 separable=False):
        super(ASPPNoGAP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ))

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = decoder.ASPPConv if not separable else decoder.ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(4 * out_channels,
                      out_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
