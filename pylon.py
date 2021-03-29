from dataclasses import dataclass

import torch
import torch.nn.functional as F
from segmentation_models_pytorch.base import (SegmentationHead,
                                              SegmentationModel)
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn
from utils.pretrain import *


@dataclass
class PylonConfig:
    backbone: str = 'resnet50'
    weights: str = 'imagenet'
    n_in: int = 1
    n_out: int = 14
    # number of decoding feature maps
    n_dec_ch: int = 128
    # number of UP modules
    n_up: int = 3
    # prediction head kernel size
    seg_kern_size: int = 1
    # whether to use pyramidal attention
    use_pa: bool = True
    # UP module's conv layers
    # '1layer' or '2layer' (default)
    up_type: str = '2layer'
    # UP module's conv kernel size
    up_kernel_size: int = 1
    # freeze?
    # 'enc' to freeze the encoder
    freeze: str = None
    # pretraining configs
    pretrain_conf: PretrainConfig = None

    @property
    def name(self):
        name = f'pylon-{self.backbone}'
        if not self.use_pa:
            name += '-nopa'
        name += f'-uptype{self.up_type}'
        if self.up_kernel_size != 1:
            name += f'-upkern{self.up_kernel_size}'
        if self.n_up != 3:
            name += f'-up{self.n_up}'
        if self.weights is not None:
            name += f'-{self.weights}'
        name += f'-dec{self.n_dec_ch}'
        if self.seg_kern_size != 1:
            name += f'-segkern{self.seg_kern_size}'
        if self.freeze is not None:
            name += f'_freeze{self.freeze}'
        if self.pretrain_conf is not None:
            name += f'_{self.pretrain_conf.name}'
        return name


class Pylon(nn.Module):
    def __init__(self, conf: PylonConfig):
        super(Pylon, self).__init__()
        self.net = PylonCore(backbone=conf.backbone,
                             n_in=conf.n_in,
                             n_out=conf.n_out,
                             weights=conf.weights,
                             n_dec_ch=conf.n_dec_ch,
                             use_pa=conf.use_pa,
                             up_type=conf.up_type,
                             up_kernel_size=conf.up_kernel_size,
                             n_up=conf.n_up,
                             seg_kern_size=conf.seg_kern_size)
        self.pool = nn.AdaptiveMaxPool2d(1)

        if conf.pretrain_conf is not None:
            load_pretrain(conf.pretrain_conf, self)

        if conf.freeze is not None:
            if conf.freeze == 'enc':
                self.net.encoder.requires_grad_(False)
            else:
                raise NotImplementedError()

    def forward(self, x):
        # enforce float32 is a good idea
        # because if the loss function involves a reduction operation
        # it would be harmful, this prevents the problem
        seg = self.net(x).float()
        pred = self.pool(seg)
        pred = torch.flatten(pred, start_dim=1)

        return {
            'pred': pred,
            'seg': seg,
        }


class PylonCore(SegmentationModel):
    def __init__(self,
                 backbone: str,
                 n_in: int,
                 n_out: int,
                 weights: str = 'imagenet',
                 n_dec_ch: int = 128,
                 use_pa: bool = True,
                 up_type: str = '2layer',
                 up_kernel_size: int = 1,
                 n_up: int = 3,
                 seg_kern_size: int = 1):
        super(PylonCore, self).__init__()

        self.encoder = get_encoder(
            backbone,
            in_channels=n_in,
            depth=5,
            weights=weights,
        )

        self.decoder = PylonDecoder(
            encoder_channels=self.encoder.out_channels,
            n_dec_ch=n_dec_ch,
            use_pa=use_pa,
            up_type=up_type,
            up_kernel_size=up_kernel_size,
            n_up=n_up,
        )

        self.segmentation_head = SegmentationHead(in_channels=n_dec_ch,
                                                  out_channels=n_out,
                                                  activation=None,
                                                  kernel_size=seg_kern_size,
                                                  upsampling=1)

        # just to comply with SegmentationModel
        self.classification_head = None
        self.initialize()


class PylonDecoder(nn.Module):
    """returns each layer of decoder
    """
    def __init__(
        self,
        encoder_channels,
        n_dec_ch: int,
        use_pa: bool = True,
        up_type: str = '2layer',
        up_kernel_size: int = 1,
        n_up: int = 3,
        upscale_mode: str = 'bilinear',
        align_corners=True,
    ):
        super(PylonDecoder, self).__init__()
        self.n_up = n_up

        self.pa = PA(
            in_channels=encoder_channels[-1],
            out_channels=n_dec_ch,
            align_corners=align_corners,
            use_pa=use_pa,
        )

        kwargs = dict(
            out_channels=n_dec_ch,
            upscale_mode=upscale_mode,
            align_corners=align_corners,
            up_type=up_type,
            kernel_size=up_kernel_size,
        )
        if n_up >= 1:
            self.up3 = UP(
                in_channels=encoder_channels[-2],
                **kwargs,
            )
        if n_up >= 2:
            self.up2 = UP(
                in_channels=encoder_channels[-3],
                **kwargs,
            )
        if n_up >= 3:
            self.up1 = UP(
                in_channels=encoder_channels[-4],
                **kwargs,
            )

    def forward(self, *features):
        bottleneck = features[-1]
        x = self.pa(bottleneck)  # 1/32
        if self.n_up >= 1:
            x = self.up3(features[-2], x)  # 1/16
        if self.n_up >= 2:
            x = self.up2(features[-3], x)  # 1/8
        if self.n_up >= 3:
            x = self.up1(features[-4], x)  # 1/4
        return x


class PA(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_pa: bool = True,
        upscale_mode='bilinear',
        align_corners=True,
    ):
        super(PA, self).__init__()

        self.upscale_mode = upscale_mode
        self.align_corners = align_corners if upscale_mode == 'bilinear' else None
        self.use_pa = use_pa

        # middle branch
        self.mid = nn.Sequential(
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ))

        # pyramid attention branch
        if use_pa:
            self.down1 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ConvBnRelu(in_channels=in_channels,
                           out_channels=1,
                           kernel_size=7,
                           stride=1,
                           padding=3))
            self.down2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ConvBnRelu(in_channels=1,
                           out_channels=1,
                           kernel_size=5,
                           stride=1,
                           padding=2))
            self.down3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ConvBnRelu(in_channels=1,
                           out_channels=1,
                           kernel_size=3,
                           stride=1,
                           padding=1))

            self.conv3 = ConvBnRelu(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
            self.conv2 = ConvBnRelu(in_channels=1,
                                    out_channels=1,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)
            self.conv1 = ConvBnRelu(in_channels=1,
                                    out_channels=1,
                                    kernel_size=7,
                                    stride=1,
                                    padding=3)

    def forward(self, x):
        upscale_parameters = dict(mode=self.upscale_mode,
                                  align_corners=self.align_corners)

        mid = self.mid(x)

        if self.use_pa:
            x1 = self.down1(x)
            x2 = self.down2(x1)
            x3 = self.down3(x2)
            x = F.interpolate(self.conv3(x3),
                              scale_factor=2,
                              **upscale_parameters)
            x = F.interpolate(self.conv2(x2) + x,
                              scale_factor=2,
                              **upscale_parameters)
            x = F.interpolate(self.conv1(x1) + x,
                              scale_factor=2,
                              **upscale_parameters)
            x = torch.mul(x, mid)
        else:
            x = mid
        return x


class UP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up_type: str = '2layer',
        kernel_size: int = 1,
        upscale_mode: str = 'bilinear',
        align_corners=True,
    ):
        super(UP, self).__init__()

        self.upscale_mode = upscale_mode
        self.align_corners = align_corners if upscale_mode == 'bilinear' else None

        if up_type == '1layer':
            self.conv1 = ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        elif up_type == '2layer':
            self.conv1 = nn.Sequential(
                ConvBnRelu(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                ConvBnRelu(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
            )
        else:
            raise NotImplementedError()

    def forward(self, x, y):
        """
        Args:
            x: low level feature
            y: high level feature
        """
        h, w = x.size(2), x.size(3)
        y_up = F.interpolate(y,
                             size=(h, w),
                             mode=self.upscale_mode,
                             align_corners=self.align_corners)
        conv = self.conv1(x)
        return y_up + conv


class ConvBnRelu(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 add_bn: bool = True,
                 add_relu: bool = True,
                 bias: bool = True,
                 interpolate: bool = False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias,
                              groups=groups)
        self.add_relu = add_relu
        self.add_bn = add_bn
        self.interpolate = interpolate
        if add_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if add_relu:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.add_bn:
            x = self.bn(x)
        if self.add_relu:
            x = self.activation(x)
        if self.interpolate:
            x = F.interpolate(x,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=True)
        return x
