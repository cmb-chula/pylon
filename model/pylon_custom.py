import torch
import torch.nn.functional as F
from segmentation_models_pytorch.base import (SegmentationHead,
                                              SegmentationModel)
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn
from trainer.start import *
from utils.pretrain import *

from .common import *


@dataclass
class PylonCustomConfig(BaseConfig):
    backbone: str = 'resnet50'
    weights: str = 'imagenet'
    n_dec_ch: int = 128
    n_in: int = 1
    n_out: int = 14
    n_up: int = 3
    only_up: Tuple[int] = (1, 2, 3)
    seg_kern_size: int = 1
    seg_norm: bool = False
    pa_global_pooling: bool = False
    use_pa: bool = True
    up_global_pooling: bool = False
    up_type: str = 'v3'
    up_kernel_size: int = 1
    extra_linear: bool = False
    dec_ch_relu: bool = True
    use_bbox_loss: bool = False
    norm_type: str = 'batchnorm'
    n_group: int = None
    pretrain_conf: PretrainConfig = None
    freeze: str = None

    @property
    def name(self):
        name = f'pylon-{self.backbone}'
        if not self.use_pa:
            name += f'-nopa'
        if self.pa_global_pooling:
            name += f'-paGAP'
        if self.up_global_pooling:
            name += f'-upGAP'
        if self.up_type != 'v1':
            name += f'-up{self.up_type}'
        if self.up_kernel_size != 1:
            name += f'-upkern{self.up_kernel_size}'
        if self.n_up != 3:
            name += f'-up{self.n_up}'
        if self.only_up != (1, 2, 3):
            name += f'-only{self.only_up}'
        if self.weights is not None:
            name += f'-{self.weights}'
        name += f'-dec{self.n_dec_ch}'
        if self.seg_kern_size != 1:
            name += f'-segkern{self.seg_kern_size}'
        if self.seg_norm:
            name += f'-segnorm'
        if not self.dec_ch_relu:
            name += f'-nodecrelu'
        if self.extra_linear:
            name += f'-linear'
        if self.norm_type != 'batchnorm':
            assert self.n_group is not None
            name += f'-{self.norm_type}{self.n_group}'
        if self.pretrain_conf is not None:
            name += f'-{self.pretrain_conf.name}'
        if self.freeze is not None:
            name += f'-freeze{self.freeze}'
        return name

    def make_model(self):
        return PylonCustom(self)


def pylon_adam_optimizer(net: 'PylonCustom', lrs: Tuple[float]):
    assert len(lrs) == 3
    return optim.Adam([
        {
            'params': net.net.encoder.parameters(),
            'lr': lrs[0]
        },
        {
            'params': net.net.decoder.parameters(),
            'lr': lrs[1]
        },
        {
            'params': net.net.segmentation_head.parameters(),
            'lr': lrs[2]
        },
    ])


class PylonCustom(nn.Module):
    def __init__(self, conf: PylonCustomConfig):
        super(PylonCustom, self).__init__()
        self.conf = conf
        self.net = PylonCore(conf)
        self.pool = nn.AdaptiveMaxPool2d(1)

        if conf.pretrain_conf is not None:
            load_pretrain(conf.pretrain_conf, target=self)

        if conf.freeze is None:
            pass
        elif conf.freeze == 'encoder':
            self.net.encoder.requires_grad_(False)
        elif conf.freeze == 'allbuthead':
            self.net.encoder.requires_grad_(False)
            self.net.decoder.requires_grad_(False)
        else:
            raise NotImplementedError()

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


class PylonCore(SegmentationModel):
    def __init__(self, conf: PylonCustomConfig):
        super(PylonCore, self).__init__()
        self.conf = conf

        self.encoder = get_encoder(
            conf.backbone,
            in_channels=conf.n_in,
            depth=5,
            weights=conf.weights,
        )

        self.decoder = PylonDecoder(
            encoder_channels=self.encoder.out_channels,
            conf=conf,
            upscale_mode='bilinear',
            align_corners=True,
        )

        if conf.seg_norm:
            self.seg_norm = nn.BatchNorm2d(conf.n_dec_ch)

        if conf.extra_linear or not conf.dec_ch_relu:
            # bn + relu + conv
            self.segmentation_head = HeadBnReluConv(
                n_in=conf.n_dec_ch,
                n_out=conf.n_out,
                kernel_size=conf.seg_kern_size)
        else:
            # just conv
            self.segmentation_head = SegmentationHead(
                in_channels=conf.n_dec_ch,
                out_channels=conf.n_out,
                activation=None,
                kernel_size=conf.seg_kern_size,
                upsampling=1)

        # just to comply with SegmentationModel
        self.classification_head = None
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        if self.conf.seg_norm:
            decoder_output = self.seg_norm(decoder_output)
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


class HeadBnReluConv(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(n_in)
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=kernel_size)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        return x


class PylonDecoder(nn.Module):
    """returns each layer of decoder
    """
    def __init__(
        self,
        encoder_channels,
        conf: PylonCustomConfig,
        upscale_mode: str = 'bilinear',
        align_corners=True,
    ):
        super(PylonDecoder, self).__init__()
        self.conf = conf
        self.pa = PA(
            in_channels=encoder_channels[-1],
            out_channels=conf.n_dec_ch,
            align_corners=align_corners,
            mid_relu=conf.dec_ch_relu,
            extra_linear=conf.extra_linear,
            use_pa=conf.use_pa,
            global_pooling=conf.pa_global_pooling,
            norm_type=conf.norm_type,
            n_group=conf.n_group,
        )

        kwargs = dict(
            out_channels=conf.n_dec_ch,
            upscale_mode=upscale_mode,
            align_corners=align_corners,
            extra_linear=conf.extra_linear,
            up_relu=conf.dec_ch_relu,
            up_type=conf.up_type,
            kernel_size=conf.up_kernel_size,
            global_pooling=conf.up_global_pooling,
            norm_type=conf.norm_type,
            n_group=conf.n_group,
        )
        if conf.n_up >= 1:
            self.up3 = UP(
                in_channels=encoder_channels[-2],
                **kwargs,
            )
        if conf.n_up >= 2:
            self.up2 = UP(
                in_channels=encoder_channels[-3],
                **kwargs,
            )
        if conf.n_up >= 3:
            self.up1 = UP(
                in_channels=encoder_channels[-4],
                **kwargs,
            )

    def forward(self, *features):
        bottleneck = features[-1]
        x = self.pa(bottleneck)  # 1/32
        if self.conf.n_up >= 1 and 1 in self.conf.only_up:
            x = self.up3(features[-2], x)  # 1/16
        if self.conf.n_up >= 2 and 2 in self.conf.only_up:
            x = self.up2(features[-3], x)  # 1/8
        if self.conf.n_up >= 3 and 3 in self.conf.only_up:
            x = self.up1(features[-4], x)  # 1/4
        return x


class PA(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        extra_linear: bool,
        mid_relu: bool,
        global_pooling: bool,
        use_pa: bool,
        upscale_mode='bilinear',
        align_corners=True,
        norm_type: str = 'batchnorm',
        n_group: int = None,
    ):
        super(PA, self).__init__()

        self.upscale_mode = upscale_mode
        self.align_corners = align_corners if upscale_mode == 'bilinear' else None
        self.extra_linear = extra_linear
        self.mid_relu = mid_relu
        self.global_pooling = global_pooling
        self.use_pa = use_pa

        # middle branch
        self.mid = nn.Sequential(
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                add_bn=mid_relu,
                add_relu=mid_relu,
                norm_type=norm_type,
                n_group=n_group,
            ))

        if extra_linear:
            self.mid2 = nn.Conv2d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=1)

        # pyramid attention branch
        if use_pa:
            self.down1 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ConvBnRelu(
                    in_channels=in_channels,
                    out_channels=1,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    norm_type=norm_type,
                    n_group=1,
                ),
            )
            self.down2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ConvBnRelu(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    norm_type=norm_type,
                    n_group=1,
                ),
            )
            self.down3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ConvBnRelu(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_type=norm_type,
                    n_group=1,
                ),
            )

            self.conv3 = ConvBnRelu(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_type=norm_type,
                n_group=1,
            )
            self.conv2 = ConvBnRelu(
                in_channels=1,
                out_channels=1,
                kernel_size=5,
                stride=1,
                padding=2,
                norm_type=norm_type,
                n_group=1,
            )
            self.conv1 = ConvBnRelu(
                in_channels=1,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_type=norm_type,
                n_group=1,
            )

        if global_pooling:
            self.branch1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvBnRelu(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_type=norm_type,
                    n_group=n_group,
                ))

        self.tmp = {}

    def forward(self, x):
        upscale_parameters = dict(mode=self.upscale_mode,
                                  align_corners=self.align_corners)

        if self.global_pooling:
            h, w = x.size(2), x.size(3)
            b = self.branch1(x)
            b = F.interpolate(b, size=(h, w), **upscale_parameters)
        else:
            b = 0

        mid = self.mid(x)
        self.tmp['mid'] = mid

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
            self.tmp['attn'] = x.clone()
            x = torch.mul(x, mid)
        else:
            x = mid
        if self.extra_linear:
            x = self.mid2(x)
        x = x + b
        return x


class UP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        extra_linear: bool,
        up_relu: bool,
        up_type: str,
        kernel_size: int,
        global_pooling: bool,
        upscale_mode: str = 'bilinear',
        align_corners=True,
        norm_type: str = 'batchnorm',
        n_group: int = None,
    ):
        super(UP, self).__init__()

        self.upscale_mode = upscale_mode
        self.align_corners = align_corners if upscale_mode == 'bilinear' else None
        self.extra_linear = extra_linear
        self.up_relu = up_relu
        self.global_pooling = global_pooling

        if up_type == 'v1':
            self.conv1 = ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                add_bn=up_relu,
                add_relu=up_relu,
                norm_type=norm_type,
                n_group=n_group,
            )
        elif up_type == 'v2':
            # deeper with n_dec_ch
            self.conv1 = nn.Sequential(
                ConvBnRelu(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    norm_type=norm_type,
                    n_group=n_group,
                ),
                ConvBnRelu(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    add_bn=up_relu,
                    add_relu=up_relu,
                    norm_type=norm_type,
                    n_group=n_group,
                ),
            )
        elif up_type == 'v3':
            # deeper with n_in ch
            self.conv1 = nn.Sequential(
                ConvBnRelu(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    norm_type=norm_type,
                    n_group=n_group,
                ),
                ConvBnRelu(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    add_bn=up_relu,
                    add_relu=up_relu,
                    norm_type=norm_type,
                    n_group=n_group,
                ),
            )
        else:
            raise NotImplementedError()

        if extra_linear:
            self.conv2 = nn.Conv2d(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=1)

        if global_pooling:
            self.conv3 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvBnRelu(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    add_relu=False,
                    norm_type=norm_type,
                    n_group=n_group,
                ), nn.Sigmoid())

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
        if self.extra_linear:
            conv = self.conv2(conv)
        if self.global_pooling:
            y = self.conv3(y)
            conv = y * conv
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
                 interpolate: bool = False,
                 norm_type: str = 'batchnorm',
                 n_group: int = None):
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
            if norm_type == 'batchnorm':
                self.bn = nn.BatchNorm2d(out_channels)
            elif norm_type == 'groupnorm':
                self.bn = nn.GroupNorm(n_group, out_channels)
            else:
                raise NotImplementedError()
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
