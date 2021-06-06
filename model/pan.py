from typing import Optional, Union

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import (ClassificationHead,
                                              SegmentationHead,
                                              SegmentationModel)
from segmentation_models_pytorch.encoders import get_encoder
from trainer.start import *

from .common import *


@dataclass
class PANConfig(BaseConfig):
    n_out: int
    backbone: str = 'resnet50'
    n_in: int = 1
    n_dec_ch: int = 128
    dilate: bool = False
    use_gap: bool = True
    weights: str = 'imagenet'

    @property
    def name(self):
        name = f'pan-{self.backbone}-dec{self.n_dec_ch}'
        if self.dilate:
            name += f'-dilate'
        if not self.use_gap:
            name += f'-nogap'
        if self.weights is not None:
            name += f'-{self.weights}'
        return name

    def make_model(self):
        return PAN(self)


class PAN(nn.Module):
    def __init__(self, conf: PANConfig):
        super().__init__()
        self.net = PANCore(conf.backbone,
                           in_channels=conf.n_in,
                           decoder_channels=conf.n_dec_ch,
                           encoder_dilation=conf.dilate,
                           upsampling=1,
                           classes=conf.n_out,
                           encoder_weights=conf.weights,
                           use_gap=conf.use_gap)
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


class PANCore(SegmentationModel):
    """ Implementation of PAN_ (Pyramid Attention Network).

    Note:
        Currently works with shape of input tensor >= [B x C x 128 x 128] for pytorch <= 1.1.0
        and with shape of input tensor >= [B x C x 256 x 256] for pytorch == 1.3.1

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_dilation: Flag to use dilation in encoder last layer. Doesn't work with ***ception***, **vgg***, 
            **densenet*`** backbones, default is **True**
        decoder_channels: A number of convolution layer filters in decoder blocks
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Avaliable options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **PAN**

    .. _PAN:
        https://arxiv.org/abs/1805.10180

    """
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 encoder_dilation: bool = True,
                 decoder_channels: int = 32,
                 in_channels: int = 3,
                 classes: int = 1,
                 activation: Optional[Union[str, callable]] = None,
                 upsampling: int = 4,
                 aux_params: Optional[dict] = None,
                 use_gap: bool = False):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        if encoder_dilation:
            self.encoder.make_dilated(stage_list=[5], dilation_list=[2])

        self.decoder = PANDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            use_gap=use_gap,
        )

        self.segmentation_head = SegmentationHead(in_channels=decoder_channels,
                                                  out_channels=classes,
                                                  activation=activation,
                                                  kernel_size=3,
                                                  upsampling=upsampling)

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "pan-{}".format(encoder_name)
        self.initialize()


class PANDecoder(nn.Module):
    def __init__(self,
                 encoder_channels,
                 decoder_channels,
                 upscale_mode: str = 'bilinear',
                 use_gap: bool = True):
        super().__init__()

        self.fpa = FPABlock(in_channels=encoder_channels[-1],
                            out_channels=decoder_channels,
                            use_gap=use_gap)
        self.gau3 = GAUBlock(in_channels=encoder_channels[-2],
                             out_channels=decoder_channels,
                             upscale_mode=upscale_mode,
                             use_gap=use_gap)
        self.gau2 = GAUBlock(in_channels=encoder_channels[-3],
                             out_channels=decoder_channels,
                             upscale_mode=upscale_mode,
                             use_gap=use_gap)
        self.gau1 = GAUBlock(in_channels=encoder_channels[-4],
                             out_channels=decoder_channels,
                             upscale_mode=upscale_mode,
                             use_gap=use_gap)

    def forward(self, *features):
        bottleneck = features[-1]
        x5 = self.fpa(bottleneck)  # 1/32
        x4 = self.gau3(features[-2], x5)  # 1/16
        x3 = self.gau2(features[-3], x4)  # 1/8
        x2 = self.gau1(features[-4], x3)  # 1/4

        return x2


class FPABlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 upscale_mode='bilinear',
                 use_gap=True):
        super(FPABlock, self).__init__()

        self.upscale_mode = upscale_mode
        self.use_gap = use_gap
        if self.upscale_mode == 'bilinear':
            self.align_corners = True
        else:
            self.align_corners = False

        if use_gap:
            # global pooling branch
            self.branch1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvBnRelu(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0))

        # midddle branch
        self.mid = nn.Sequential(
            ConvBnRelu(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=1,
                       stride=1,
                       padding=0))
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
                       padding=1),
            ConvBnRelu(in_channels=1,
                       out_channels=1,
                       kernel_size=3,
                       stride=1,
                       padding=1),
        )
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
        h, w = x.size(2), x.size(3)
        upscale_parameters = dict(mode=self.upscale_mode,
                                  align_corners=self.align_corners)
        if self.use_gap:
            b1 = self.branch1(x)
            b1 = F.interpolate(b1, size=(h, w), **upscale_parameters)

        mid = self.mid(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = F.interpolate(x3, size=(h // 4, w // 4), **upscale_parameters)

        x2 = self.conv2(x2)
        x = x2 + x3
        x = F.interpolate(x, size=(h // 2, w // 2), **upscale_parameters)

        x1 = self.conv1(x1)
        x = x + x1
        x = F.interpolate(x, size=(h, w), **upscale_parameters)

        x = torch.mul(x, mid)
        if self.use_gap:
            x = x + b1
        return x


class GAUBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 upscale_mode: str = 'bilinear',
                 use_gap: bool = True):
        super(GAUBlock, self).__init__()

        self.upscale_mode = upscale_mode
        self.use_gap = use_gap
        self.align_corners = True if upscale_mode == 'bilinear' else None

        if use_gap:
            self.conv1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvBnRelu(in_channels=out_channels,
                           out_channels=out_channels,
                           kernel_size=1,
                           add_relu=False), nn.Sigmoid())
        self.conv2 = ConvBnRelu(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=1)

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
        x = self.conv2(x)
        if self.use_gap:
            y = self.conv1(y)
            z = torch.mul(x, y)
        else:
            z = x
        return y_up + z


class ConvBnRelu(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 add_relu: bool = True,
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
        self.interpolate = interpolate
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.add_relu:
            x = self.activation(x)
        if self.interpolate:
            x = F.interpolate(x,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=True)
        return x
