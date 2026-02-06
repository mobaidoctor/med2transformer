# +
import os
import os.path as osp
import random
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.utils.data as data
from os.path import join
from collections import OrderedDict
from torchvision import models
import SimpleITK as sitk
import pydicom
import scipy.io
import glob

from torch.utils.data import Dataset

from kornia.augmentation import (
    RandomHorizontalFlip3D,
    RandomVerticalFlip3D,
)

from pytorch_wavelets import DWTForward
from timm.models.layers import DropPath, trunc_normal_
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

bias_setting = False


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def make_norm_3d(norm_layer, num_features):
    """Instantiate a 3D norm layer from either a class or functools.partial."""
    if norm_layer is None:
        return nn.Identity()
    return norm_layer(num_features)


class ShareSepConv3d(nn.Module):
    """
    Shared Separable Convolution in 3D.

    This module creates a convolution weight tensor (with a center element of 1)
    and applies a 3D convolution with the weight shared across input channels.
    """
    def __init__(self, kernel_size):
        super(ShareSepConv3d, self).__init__()
        assert kernel_size % 2 == 1, 'Kernel size should be odd'
        self.padding = (kernel_size - 1) // 2
        weight_tensor = torch.zeros(1, 1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        # Expand the weight to match the number of input channels.
        expand_weight = self.weight.expand(inc, 1, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv3d(
            x,
            expand_weight,
            None,
            stride=1,
            padding=(0, self.padding, self.padding),
            dilation=1,
            groups=inc,
        )


class SmoothDilatedResidualBlock3d(nn.Module):
    """
    A 3D residual block with smooth dilated convolutions.
    """
    def __init__(self, dim, dilation, norm_layer, activation=nn.ReLU(True)):
        super(SmoothDilatedResidualBlock3d, self).__init__()

        conv_block = [
            ShareSepConv3d(dilation[1] * 2 - 1),
            nn.Conv3d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            make_norm_3d(norm_layer, dim),
            activation,
            ShareSepConv3d(dilation[1] * 2 - 1),
            nn.Conv3d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            make_norm_3d(norm_layer, dim),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Pix2pix_3d(nn.Module):
    """
    A 3D Pix2Pix generator network.
    """
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=16,
        n_downsampling=3,
        n_blocks=9,
        norm_layer=nn.InstanceNorm3d,
        padding_type='zero',
        resblock_type='smoothdilated',
        upsample_type='nearest',
        skip_connection=True,
    ):
        assert n_blocks >= 0, "Number of residual blocks must be non-negative"
        super(Pix2pix_3d, self).__init__()

        self.skip_connection = skip_connection
        activation = nn.ReLU(True)

        conv0 = [
            nn.Conv3d(input_nc, ngf, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False),
            make_norm_3d(norm_layer, ngf),
            activation,
        ]
        self.conv0 = nn.Sequential(*conv0)

        mult = 1
        conv_down1 = [
            nn.Conv3d(
                ngf * mult,
                ngf * mult * 2,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            make_norm_3d(norm_layer, ngf * mult * 2),
            activation,
        ]
        self.conv_down1 = nn.Sequential(*conv_down1)

        mult = 2
        conv_down2 = [
            nn.Conv3d(
                ngf * mult,
                ngf * mult * 2,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            make_norm_3d(norm_layer, ngf * mult * 2),
            activation,
        ]
        self.conv_down2 = nn.Sequential(*conv_down2)

        mult = 4
        conv_down3 = [
            nn.Conv3d(
                ngf * mult,
                ngf * mult * 2,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False,
            ),
            make_norm_3d(norm_layer, ngf * mult * 2),
            activation,
        ]
        self.conv_down3 = nn.Sequential(*conv_down3)

        mult = 8
        convt_up3 = [
            nn.Upsample(scale_factor=(1, 2, 2), mode=upsample_type),
            nn.Conv3d(
                ngf * mult,
                int(ngf * mult / 2),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            make_norm_3d(norm_layer, int(ngf * mult / 2)),
            activation,
        ]
        self.convt_up3 = nn.Sequential(*convt_up3)

        mult = 4
        in_channels = ngf * mult * 2 if skip_connection else ngf * mult
        decoder_conv3 = [
            nn.Conv3d(
                in_channels,
                ngf * mult,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            make_norm_3d(norm_layer, ngf * mult),
            activation,
        ]
        self.decoder_conv3 = nn.Sequential(*decoder_conv3)

        mult = 4
        convt_up2 = [
            nn.Upsample(scale_factor=(1, 2, 2), mode=upsample_type),
            nn.Conv3d(
                ngf * mult,
                int(ngf * mult / 2),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            make_norm_3d(norm_layer, int(ngf * mult / 2)),
            activation,
        ]
        self.convt_up2 = nn.Sequential(*convt_up2)

        mult = 2
        in_channels = ngf * mult * 2 if skip_connection else ngf * mult
        decoder_conv2 = [
            nn.Conv3d(
                in_channels,
                ngf * mult,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            ),
            make_norm_3d(norm_layer, ngf * mult),
            activation,
        ]
        self.decoder_conv2 = nn.Sequential(*decoder_conv2)

        mult = 2
        convt_up1 = [
            nn.Upsample(scale_factor=(2, 2, 2), mode=upsample_type),
            nn.Conv3d(
                ngf * mult,
                int(ngf * mult / 2),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            make_norm_3d(norm_layer, int(ngf * mult / 2)),
            activation,
        ]
        self.convt_up1 = nn.Sequential(*convt_up1)

        in_channels = ngf * 2 if skip_connection else ngf
        decoder_conv1 = [
            nn.Conv3d(
                in_channels,
                output_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.Tanh(),
        ]
        self.decoder_conv1 = nn.Sequential(*decoder_conv1)

    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv_down1(x0)
        x2 = self.conv_down2(x1)
        x3 = self.conv_down3(x2)

        x4 = self.convt_up3(x3)
        if self.skip_connection:
            x4 = torch.cat((x4, x2), dim=1)
        x4 = self.decoder_conv3(x4)

        x5 = self.convt_up2(x4)
        if self.skip_connection:
            x5 = torch.cat((x5, x1), dim=1)
        x5 = self.decoder_conv2(x5)

        x6 = self.convt_up1(x5)
        if self.skip_connection:
            x6 = torch.cat((x6, x0), dim=1)
        out = self.decoder_conv1(x6)
        return out


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
            classname.find('Conv') != -1 or classname.find('Linear') != -1
        ) and (classname != 'ShareSepConv3d'):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            print('Norm initialized')
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=None):
    """
    Initialize a network:
      1. register CPU/GPU device (with multi-GPU support)
      2. initialize the network weights.

    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- normal | xavier | kaiming | orthogonal
        init_gain (float)  -- scaling factor
        gpu_ids (list[int])-- which GPUs the network runs on: e.g., [0,1,2]

    Return an initialized (and possibly DataParallel-wrapped) network.
    """
    if gpu_ids is None:
        gpu_ids = []
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available(), "CUDA is not available but gpu_ids are set."
        device = torch.device(f'cuda:{gpu_ids[0]}')
        net.to(device)
        if len(gpu_ids) > 1:
            net = nn.DataParallel(net, device_ids=gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(
    input_nc,
    output_nc,
    ngf,
    resolution,
    netG,
    norm='batch',
    use_dropout=False,
    init_type='normal',
    init_gain=0.02,
    gpu_ids=None,
):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'Med2Transformer':
        net = Med2Transformer(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            upsample_type='nearest',
            skip_connection=True,
            resolution=resolution,
        )
    elif netG == 'pix2pix3d':
        net = Pix2pix_3d(
            input_nc,
            output_nc,
            ngf=ngf,
            n_downsampling=3,
            n_blocks=9,
            norm_layer=make_norm_3d if norm_layer is None else norm_layer.func if isinstance(norm_layer, functools.partial) else norm_layer,
        )
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids or [])


def define_D(
    input_nc,
    ndf,
    netD,
    n_layers_D=3,
    norm='batch',
    init_type='normal',
    init_gain=0.02,
    gpu_ids=None,
):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | wave3DDiscriminator
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., [0,1,2]

    Returns:
        A discriminator network.
    """
    gpu_ids = gpu_ids or []
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayer3DDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayer3DDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'wave3DDiscriminator':  # wavelet-based 3D discriminator
        net = wave3DDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


class Med2Transformer(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=16,
        n_downsampling=3,
        n_blocks=9,
        norm_layer=nn.InstanceNorm3d,
        padding_type="zero",
        resblock_type="smoothdilated",
        upsample_type="nearest",
        skip_connection=True,
        resolution=None,
        num_classes: int = 3,
    ):
        assert (n_blocks >= 0)
        super(Med2Transformer, self).__init__()

        self.skip_connection = skip_connection
        activation = nn.ReLU(True)

        conv0 = [
            nn.Conv3d(
                input_nc,
                ngf,
                kernel_size=(3, 5, 5),
                padding=(1, 2, 2),
                bias=bias_setting,
            ),
            make_norm_3d(norm_layer, ngf),
            activation,
        ]
        self.conv0 = nn.Sequential(*conv0)

        mult = 1
        conv_down1 = [
            nn.Conv3d(
                ngf * mult,
                ngf * mult * 2,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                bias=bias_setting,
            ),
            make_norm_3d(norm_layer, ngf * mult * 2),
            activation,
        ]
        self.conv_down1 = nn.Sequential(*conv_down1)

        mult = 2
        conv_down2 = [
            nn.Conv3d(
                ngf * mult,
                ngf * mult * 2,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                bias=bias_setting,
            ),
            make_norm_3d(norm_layer, ngf * mult * 2),
            activation,
        ]
        self.conv_down2 = nn.Sequential(*conv_down2)

        mult = 4
        conv_down3 = [
            nn.Conv3d(
                ngf * mult,
                ngf * mult * 2,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=bias_setting,
            ),
            make_norm_3d(norm_layer, ngf * mult * 2),
            activation,
        ]
        self.conv_down3 = nn.Sequential(*conv_down3)

        mult = 8
        head_swin = [8]

        aggregate1 = [
            nn.Conv3d(ngf * 2 * 2, ngf * 2, kernel_size=1, stride=1, bias=bias_setting),
            make_norm_3d(norm_layer, ngf * 2),
            activation,
        ]
        self.aggregate1 = nn.Sequential(*aggregate1)
        self.mixall_1 = nn.Conv3d(
            in_channels=ngf * 2,
            out_channels=ngf * 2,
            kernel_size=1,
            stride=1,
        )

        aggregate2 = [
            nn.Conv3d(ngf * 4 * 2, ngf * 4, kernel_size=1, stride=1, bias=bias_setting),
            make_norm_3d(norm_layer, ngf * 4),
            activation,
        ]
        self.aggregate2 = nn.Sequential(*aggregate2)
        self.mixall_2 = nn.Conv3d(
            in_channels=ngf * 4,
            out_channels=ngf * 4,
            kernel_size=1,
            stride=1,
        )

        aggregate3 = [
            nn.Conv3d(ngf * 8 * 2, ngf * 8, kernel_size=1, stride=1, bias=bias_setting),
            make_norm_3d(norm_layer, ngf * 8),
            activation,
        ]
        self.aggregate3 = nn.Sequential(*aggregate3)
        self.mixall_3 = nn.Conv3d(
            in_channels=ngf * 8,
            out_channels=ngf * 8,
            kernel_size=1,
            stride=1,
        )

        self.MSwinBlock1 = MSwintransformer(
            ngf * mult,
            dilation=[1, 1, 1],
            activation=activation,
            norm_layer=norm_layer,
            n_downsampling=n_downsampling,
            last_window_size=[[2, 4, 4]],
            last_num_heads=head_swin,
            num_layer=3,
            resolution=resolution,
        )
        self.MSwinBlock2 = MSwintransformer(
            ngf * mult,
            dilation=[1, 1, 1],
            activation=activation,
            norm_layer=norm_layer,
            n_downsampling=n_downsampling,
            last_window_size=[[2, 4, 4]],
            last_num_heads=head_swin,
            num_layer=3,
            resolution=resolution,
        )
        self.MSwinBlock3 = MSwintransformer(
            ngf * mult,
            dilation=[1, 1, 1],
            activation=activation,
            norm_layer=norm_layer,
            n_downsampling=n_downsampling,
            last_window_size=[[2, 4, 4]],
            last_num_heads=head_swin,
            num_layer=2,
            resolution=resolution,
        )
        self.MSwinBlock4 = MSwintransformer(
            ngf * mult,
            dilation=[1, 2, 2],
            activation=activation,
            norm_layer=norm_layer,
            n_downsampling=n_downsampling,
            last_window_size=[[2, 4, 4]],
            last_num_heads=head_swin,
            num_layer=2,
            resolution=resolution,
        )
        self.MSwinBlock5 = MSwintransformer(
            ngf * mult,
            dilation=[1, 2, 2],
            activation=activation,
            norm_layer=norm_layer,
            n_downsampling=n_downsampling,
            last_window_size=[[2, 4, 4]],
            last_num_heads=head_swin,
            num_layer=1,
            resolution=resolution,
        )
        self.MSwinBlock6 = MSwintransformer(
            ngf * mult,
            dilation=[1, 2, 2],
            activation=activation,
            norm_layer=norm_layer,
            n_downsampling=n_downsampling,
            last_window_size=[[2, 4, 4]],
            last_num_heads=head_swin,
            num_layer=1,
            resolution=resolution,
        )

        self.Branch_MSwin_1 = MSwintransformer(
            int(ngf * mult / 4),
            dilation=[1, 1, 1],
            activation=activation,
            norm_layer=norm_layer,
            n_downsampling=n_downsampling - 2,
            last_window_size=[[2, 4, 4]],
            last_num_heads=head_swin,
            num_layer=1,
            resolution=resolution,
        )
        self.Branch_MSwin_2 = MSwintransformer(
            int(ngf * mult / 4),
            dilation=[1, 2, 2],
            activation=activation,
            norm_layer=norm_layer,
            n_downsampling=n_downsampling - 2,
            last_window_size=[[2, 4, 4]],
            last_num_heads=head_swin,
            num_layer=1,
            resolution=resolution,
        )

        self.Branch_MSwin_3 = MSwintransformer(
            int(ngf * mult / 2),
            dilation=[1, 1, 1],
            activation=activation,
            norm_layer=norm_layer,
            n_downsampling=n_downsampling - 1,
            last_window_size=[[2, 4, 4]],
            last_num_heads=head_swin,
            num_layer=1,
            resolution=resolution,
        )
        self.Branch_MSwin_4 = MSwintransformer(
            int(ngf * mult / 2),
            dilation=[1, 2, 2],
            activation=activation,
            norm_layer=norm_layer,
            n_downsampling=n_downsampling - 1,
            last_window_size=[[2, 4, 4]],
            last_num_heads=head_swin,
            num_layer=1,
            resolution=resolution,
        )

        self.Branch_MSwin_5 = MSwintransformer(
            ngf * mult,
            dilation=[1, 1, 1],
            activation=activation,
            norm_layer=norm_layer,
            n_downsampling=n_downsampling,
            last_window_size=[[2, 4, 4]],
            last_num_heads=head_swin,
            num_layer=1,
            resolution=resolution,
        )
        self.Branch_MSwin_6 = MSwintransformer(
            ngf * mult,
            dilation=[1, 2, 2],
            activation=activation,
            norm_layer=norm_layer,
            n_downsampling=n_downsampling,
            last_window_size=[[2, 4, 4]],
            last_num_heads=head_swin,
            num_layer=1,
            resolution=resolution,
        )

        mult = 8
        convt_up3 = [
            nn.Upsample(scale_factor=(1, 2, 2), mode=upsample_type),
            nn.Conv3d(
                ngf * mult,
                int(ngf * mult / 2),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias_setting,
            ),
            make_norm_3d(norm_layer, int(ngf * mult / 2)),
            activation,
        ]
        self.convt_up3 = nn.Sequential(*convt_up3)

        mult = 4
        if skip_connection:
            in_channels = ngf * mult * 2
        else:
            in_channels = ngf * mult
        decoder_conv3 = [
            nn.Conv3d(
                in_channels,
                ngf * mult,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias_setting,
            ),
            make_norm_3d(norm_layer, ngf * mult),
            activation,
        ]
        self.decoder_conv3 = nn.Sequential(*decoder_conv3)

        mult = 4
        convt_up2 = [
            nn.Upsample(scale_factor=(1, 2, 2), mode=upsample_type),
            nn.Conv3d(
                ngf * mult,
                int(ngf * mult / 2),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias_setting,
            ),
            make_norm_3d(norm_layer, int(ngf * mult / 2)),
            activation,
        ]
        self.convt_up2 = nn.Sequential(*convt_up2)

        mult = 2
        if skip_connection:
            in_channels = ngf * mult * 2
        else:
            in_channels = ngf * mult
        decoder_conv2 = [
            nn.Conv3d(
                in_channels,
                ngf * mult,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=bias_setting,
            ),
            make_norm_3d(norm_layer, ngf * mult),
            activation,
        ]
        self.decoder_conv2 = nn.Sequential(*decoder_conv2)

        mult = 2
        convt_up1 = [
            nn.Upsample(scale_factor=(2, 2, 2), mode=upsample_type),
            nn.Conv3d(
                ngf * mult,
                int(ngf * mult / 2),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias_setting,
            ),
            make_norm_3d(norm_layer, int(ngf * mult / 2)),
            activation,
        ]
        self.convt_up1 = nn.Sequential(*convt_up1)

        if skip_connection:
            in_channels = ngf * 2
        else:
            in_channels = ngf
        decoder_conv1 = [
            nn.Conv3d(
                in_channels,
                output_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.Tanh(),
        ]
        self.decoder_conv1 = nn.Sequential(*decoder_conv1)

        self.patch_embed_1 = project(1, 32, [2, 2, 2], [2, 2, 2], nn.GELU, nn.LayerNorm)
        self.patch_embed_2 = project(1, 64, [2, 4, 4], [2, 4, 4], nn.GELU, nn.LayerNorm)
        self.patch_embed_3 = project(1, 128, [2, 8, 8], [2, 8, 8], nn.GELU, nn.LayerNorm)

    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv_down1(x0)

        # Patch embedding 1 + branch Swin
        x11 = self.patch_embed_1(input)
        x11a = self.Branch_MSwin_1(x11)
        x11b = self.Branch_MSwin_2(x11)
        x1s_concat = torch.cat([x11a, x11b], dim=1)
        x1s = self.aggregate1(x1s_concat)
        x11 = self.mixall_1(x1s) + x1s

        x1_concat = torch.cat([x11, x1], dim=1)
        x1 = self.aggregate1(x1_concat)
        x1 = self.mixall_1(x1) + x1

        x2 = self.conv_down2(x1)

        # Patch embedding 2 + branch Swin
        x21 = self.patch_embed_2(input)
        x21a = self.Branch_MSwin_3(x21)
        x21b = self.Branch_MSwin_4(x21)
        x2s_concat = torch.cat([x21a, x21b], dim=1)
        x2s = self.aggregate2(x2s_concat)
        x21 = self.mixall_2(x2s) + x2s

        x2_concat = torch.cat([x21, x2], dim=1)
        x2 = self.aggregate2(x2_concat)
        x2 = self.mixall_2(x2) + x2

        x3 = self.conv_down3(x2)

        # Patch embedding 3 + deeper Swin
        x31 = self.patch_embed_3(input)
        x31a = self.Branch_MSwin_5(x31)
        x31b = self.Branch_MSwin_6(x31)
        x3s_concat = torch.cat([x31a, x31b], dim=1)
        x3s = self.aggregate3(x3s_concat)
        x31 = self.mixall_3(x3s) + x3s

        x3_concat = torch.cat([x31, x3], dim=1)
        x3 = self.aggregate3(x3_concat)
        x3 = self.mixall_3(x3) + x3

        # Deep Swin transformer parallel branches
        x3a = self.MSwinBlock1(x3)
        x3a = self.MSwinBlock2(x3a)
        x3a = self.MSwinBlock3(x3a)

        x3b = self.MSwinBlock4(x3)
        x3b = self.MSwinBlock3(x3b)
        x3b = self.MSwinBlock4(x3b)

        x3c_concat = torch.cat([x3a, x3b], dim=1)
        x3 = self.aggregate3(x3c_concat)
        x3 = self.mixall_3(x3) + x3

        # Decoder + skip connections
        x4 = self.convt_up3(x3)
        if self.skip_connection:
            x4 = torch.cat((x4, x2), dim=1)
        x4 = self.decoder_conv3(x4)

        x5 = self.convt_up2(x4)
        if self.skip_connection:
            x5 = torch.cat((x5, x1), dim=1)
        x5 = self.decoder_conv2(x5)

        x6 = self.convt_up1(x5)
        if self.skip_connection:
            x6 = torch.cat((x6, x0), dim=1)
        out = self.decoder_conv1(x6)

        return out


class ResnetBlock3d(nn.Module):
    """
    A basic 3D ResNet block.
    """
    def __init__(self, dim, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock3d, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, norm_layer, activation, use_dropout):
        conv_block = []
        p = 1
        conv_block += [
            nn.Conv3d(dim, dim, kernel_size=3, padding=p),
            make_norm_3d(norm_layer, dim),
            activation,
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [
            nn.Conv3d(dim, dim, kernel_size=3, padding=p),
            make_norm_3d(norm_layer, dim),
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class project(nn.Module):
    """
    A module for patch embedding projection.
    """
    def __init__(self, in_dim, out_dim, kernel_size, stride, activate, norm, last=False):
        super(project, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv3d(in_dim, out_dim // 2, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv3d(out_dim // 2, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim // 2)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim // 2, Ws, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Ws, Wh, Ww)
        return x


class NLayer3DDiscriminator(nn.Module):
    """
    Defines a 3D PatchGAN discriminator.
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        super(NLayer3DDiscriminator, self).__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 3
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                make_norm_3d(norm_layer, ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            make_norm_3d(norm_layer, ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input, isDetach):
        return self.model(input)


class wave3DDiscriminator(nn.Module):
    """
    A 3D discriminator that uses wavelet transforms
    to better capture local frequency content (good for CT details).
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        super(wave3DDiscriminator, self).__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.xfm = DWTForward(J=1, mode='zero', wave='haar')

        kw = 3
        padw = int(np.ceil((kw - 1) / 2))

        input_sequence = [
            nn.Conv3d(
                input_nc,
                input_nc,
                kernel_size=kw,
                stride=(1, 2, 2),
                padding=padw,
                bias=use_bias,
            ),
            make_norm_3d(norm_layer, input_nc),
            nn.LeakyReLU(0.2, True),
        ]
        self.input_model = nn.Sequential(*input_sequence)

        sequence = [
            nn.Conv3d(input_nc * 5, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                make_norm_3d(norm_layer, ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            make_norm_3d(norm_layer, ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input, isDetach=False):
        T, C, D, H, W = input.size()
        device = input.device
        
        x_wave = torch.zeros(
            [T, 5 * C, D, H // 2, W // 2],
            device=device,
            requires_grad=not isDetach,
        )

        x_wave = x_wave.clone()

        for D_idx in range(D):
            Yl, Yh = self.xfm(input[:, :, D_idx, :, :])

            x_wave[:, 0 * C:1 * C, D_idx] = Yl
            x_wave[:, 1 * C:2 * C, D_idx] = Yh[0][:, :, 0]
            x_wave[:, 2 * C:3 * C, D_idx] = Yh[0][:, :, 1]
            x_wave[:, 3 * C:4 * C, D_idx] = Yh[0][:, :, 2]

        x_wave[:, 4 * C:5 * C] = self.input_model(input)

        return self.model(x_wave)


class MSwintransformer(nn.Module):
    """
    A multi-scale swin transformer block with a convolutional sub-block.
    """
    def __init__(
        self,
        dim,
        dilation,
        norm_layer,
        n_downsampling,
        last_window_size,
        last_num_heads,
        num_layer,
        resolution,
        activation=nn.ReLU(True),
    ):
        super(MSwintransformer, self).__init__()
        conv_block = [
            ShareSepConv3d(dilation[1] * 2 - 1),
            nn.Conv3d(
                dim,
                dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=bias_setting,
            ),
            make_norm_3d(norm_layer, dim),
            activation,
            ShareSepConv3d(dilation[1] * 2 - 1),
            nn.Conv3d(
                dim,
                dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=bias_setting,
            ),
            make_norm_3d(norm_layer, dim),
        ]
        self.conv_block = nn.Sequential(*conv_block)

        depths = [2]
        self.pos_drop = nn.Dropout(p=0.0)

        self.last_hidden_size = dim
        self.change_dim = dim * 2
        self.last_window_size = last_window_size
        self.last_num_heads = last_num_heads
        self.last_num_layers = num_layer

        self.last_patch_embeddings = nn.Conv3d(
            in_channels=self.last_hidden_size,
            out_channels=self.last_hidden_size,
            kernel_size=1,
            stride=1,
        )

        self.lastlayers = nn.ModuleList()
        for i_layer in range(self.last_num_layers):
            layer = BasicLayer(
                dim=self.last_hidden_size,
                input_resolution=(
                    int(resolution[0] / 2),
                    int(resolution[1] / (2 ** (n_downsampling))),
                    int(resolution[2] / (2 ** (n_downsampling))),
                ),
                depth=depths[0],
                num_heads=last_num_heads[0],
                window_size=last_window_size[0],
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0,
                drop_path=[x.item() for x in torch.linspace(0, 0.05, sum(depths))],
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False,
                i_layer=i_layer,
            )
            self.lastlayers.append(layer)
        self.last_norm_layer = nn.LayerNorm(self.last_hidden_size)

    def forward(self, x):
        x_orgin = x
        x_conv = self.conv_block(x)
        x = x + x_conv

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = self.last_patch_embeddings(x)
        x = x.flatten(2).permute(0, 2, 1)  # (B, n_patch, hidden)
        x = self.pos_drop(x)

        x_shout = x
        for i in range(self.last_num_layers):
            layer = self.lastlayers[i]
            x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
            x = self.last_norm_layer(x)
            x = x + x_shout
            x_shout = x

        x = x.permute(0, 2, 1)
        x = x.view(-1, self.last_hidden_size, Ws, Wh, Ww).contiguous()

        out = x + x_orgin + x_conv
        return out


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    """
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=True,
        use_checkpoint=False,
        i_layer=None,
    ):
        super(BasicLayer, self).__init__()
        self.window_size = window_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.i_layer = i_layer
        self.dim = dim

        self.shift_size = [
            window_size[0] // 2,
            window_size[1] // 2,
            window_size[2] // 2,
        ]
        self.block11 = nn.ModuleList([
            Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
        ])

        self.dim_mix = nn.Linear(dim, dim)

        self.window_size_total = []
        self.window_size_total.append(
            [self.window_size[0], self.window_size[1] // 2, self.window_size[2] * 2]
        )
        self.window_size_total.append(
            [self.window_size[0], self.window_size[1] * 2, self.window_size[2] // 2]
        )

        self.shift_size_total = []
        self.shift_size_total.append(
            [
                self.window_size_total[0][0] // 2,
                self.window_size_total[0][1] // 2,
                self.window_size_total[0][2] // 2,
            ]
        )
        self.shift_size_total.append(
            [
                self.window_size_total[1][0] // 2,
                self.window_size_total[1][1] // 2,
                self.window_size_total[1][2] // 2,
            ]
        )

        self.dim_change = [0.5, 0.5]

        self.block1 = nn.ModuleList([
            Block(
                dim=int(dim * self.dim_change[0]),
                input_resolution=input_resolution,
                num_heads=int(num_heads * self.dim_change[0]),
                window_size=self.window_size_total[0],
                shift_size=self.shift_size_total[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[1] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
        ])

        self.block2 = nn.ModuleList([
            Block(
                dim=int(dim * self.dim_change[1]),
                input_resolution=input_resolution,
                num_heads=int(num_heads * self.dim_change[1]),
                window_size=self.window_size_total[1],
                shift_size=self.shift_size_total[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[1] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
        ])

        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = None

    def forward(self, x, S, H, W):
        Sp = int(np.ceil(S / self.window_size[0])) * self.window_size[0]
        Hp = int(np.ceil(H / self.window_size[1])) * self.window_size[1]
        Wp = int(np.ceil(W / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)
        s_slices = (
            slice(0, -self.window_size[0]),
            slice(-self.window_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        h_slices = (
            slice(0, -self.window_size[1]),
            slice(-self.window_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )
        w_slices = (
            slice(0, -self.window_size[2]),
            slice(-self.window_size[2], -self.shift_size[2]),
            slice(-self.shift_size[2], None),
        )
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(
            -1,
            self.window_size[0] * self.window_size[1] * self.window_size[2],
        )
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0,
            float(0.0),
        )
        for blk in self.block11:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)

        x_total = []
        for idx in range(2):
            window_size = self.window_size_total[idx]
            shift_size = self.shift_size_total[idx]
            if idx == 0:
                block = self.block1
                x_this = x[:, :, 0:int(self.dim * self.dim_change[0])]
            else:
                block = self.block2
                x_this = x[:, :, int(self.dim * self.dim_change[0]):]

            Sp = int(np.ceil(S / window_size[0])) * window_size[0]
            Hp = int(np.ceil(H / window_size[1])) * window_size[1]
            Wp = int(np.ceil(W / window_size[2])) * window_size[2]
            img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)
            s_slices = (
                slice(0, -window_size[0]),
                slice(-window_size[0], -shift_size[0]),
                slice(-shift_size[0], None),
            )
            h_slices = (
                slice(0, -window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None),
            )
            w_slices = (
                slice(0, -window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            )
            cnt = 0
            for s in s_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, s, h, w, :] = cnt
                        cnt += 1

            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(
                -1,
                window_size[0] * window_size[1] * window_size[2],
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0,
                float(0.0),
            )
            for blk in block:
                blk.H, blk.W = H, W
                x_this = blk(x_this, attn_mask)
            x_total.append(x_this)

        x = torch.cat([x_total[0], x_total[1]], dim=2)
        x = self.dim_mix(x)

        return x, S, H, W


class Block(nn.Module):
    """
    Swin Transformer Block with relative positional attention.
    """
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super(Block, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if tuple(self.input_resolution) == tuple(self.window_size):
            self.shift_size = [0, 0, 0]
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        S, H, W = self.input_resolution
        assert L == S * H * W, "Input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)

        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_g = (self.window_size[0] - S % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        _, Sp, Hp, Wp, _ = x.shape

        if isinstance(self.shift_size, list) and min(self.shift_size) > 0:
            shifted_x = torch.roll(
                x,
                shifts=(
                    -self.shift_size[0],
                    -self.shift_size[1],
                    -self.shift_size[2],
                ),
                dims=(1, 2, 3),
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(
            -1,
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            C,
        )
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(
            -1,
            self.window_size[0],
            self.window_size[1],
            self.window_size[2],
            C,
        )
        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            Sp,
            Hp,
            Wp,
        )

        if isinstance(self.shift_size, list) and min(self.shift_size) > 0:
            x = torch.roll(
                shifted_x,
                shifts=(
                    self.shift_size[0],
                    self.shift_size[1],
                    self.shift_size[2],
                ),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()
        x = x.view(B, S * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    """
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
    ):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1)
                * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1),
                num_heads,
            )
        )
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_coords[:, :, 0] *= 2 * (self.window_size[1] + self.window_size[2]) - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(
            B_,
            N,
            3,
            self.num_heads,
            C // self.num_heads,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0]
            * self.window_size[1]
            * self.window_size[2],
            self.window_size[0]
            * self.window_size[1]
            * self.window_size[2],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(
                B_ // nW,
                nW,
                self.num_heads,
                N,
                N,
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Partition input tensor into non-overlapping windows.
    """
    B, S, H, W, C = x.shape
    x = x.view(
        B,
        S // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(
        -1,
        window_size[0],
        window_size[1],
        window_size[2],
        C,
    )
    return windows


def window_reverse(windows, window_size, S, H, W):
    """
    Reverse the window partition operation.
    """
    B = int(windows.shape[0] / (S * H * W / (window_size[0] * window_size[1] * window_size[2])))
    x = windows.view(
        B,
        S // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(
        B,
        S,
        H,
        W,
        -1,
    )
    return x


class Mlp(nn.Module):
    """
    Multilayer Perceptron.
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.,
    ):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VGGLoss(nn.Module):
    """
    VGG-based perceptual loss.
    """
    def __init__(
        self,
        pretrained_dir='',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19(pretrained_dir, device=device).to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class Vgg19(nn.Module):
    """
    VGG19 network for perceptual loss.
    """
    def __init__(
        self,
        pretrained_dir='',
        requires_grad=False,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        super(Vgg19, self).__init__()
        if pretrained_dir != '':
            model_vgg = models.vgg19(pretrained=False)
            state = torch.load(pretrained_dir, map_location=device)
            model_vgg.load_state_dict(state)
            print('Successful download of pre-trained model from %s' % pretrained_dir)
            vgg_pretrained_features = model_vgg.features
        else:
            model_vgg = models.vgg19(pretrained=True)
            vgg_pretrained_features = model_vgg.features

        vgg_pretrained_features = vgg_pretrained_features.to(device)
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss_3D(nn.Module):
    """
    3D version of the VGG perceptual loss.
    """
    def __init__(self, pretrained_dir='', device=None):
        super(VGGLoss_3D, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vggloss = VGGLoss(pretrained_dir, device=device)

    def forward(self, x, y):
        _, _, n, _, _ = x.size()
        loss = 0
        for i in range(n):
            loss += self.vggloss(
                x[:, :, i, :, :].repeat(1, 3, 1, 1),
                y[:, :, i, :, :].repeat(1, 3, 1, 1),
            )
        return loss / n


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - vanilla | lsgan | wgangp
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input."""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels."""
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class L1_Charbonnier_loss(nn.Module):
    """
    L1 Charbonnier Loss.
    """
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class GANclass(nn.Module):
    """
    The GAN class wraps both generator and discriminator, defines loss functions,
    and implements the training step.
    """
    def __init__(self, opt):
        super(GANclass, self).__init__()
        self.isTrain = opt.isTrain
        self.max_epoch = opt.max_epochs
        self.resolution = [opt.depthSize, opt.ImageSize, opt.ImageSize]
        self.VGG_loss = opt.VGG_loss
        self.gpu_ids = opt.gpu_ids
        self.device = opt.device
        self.lambda_L1 = opt.lambda_L1
        self.G_model = opt.G_model

        # Losses
        if self.VGG_loss:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_loss', 'G_perceive']
        else:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_loss']

        # Networks
        self.netG = define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            self.resolution,
            opt.G_model,
            opt.G_norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        if self.isTrain:
            self.netD = define_D(
                opt.input_nc + opt.output_nc,
                opt.ndf,
                opt.D_model,
                opt.n_layers_D,
                opt.D_norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )

        if self.isTrain:
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = nn.L1Loss().to(self.device)
            self.BCELoss = nn.BCELoss().to(self.device)
            if self.VGG_loss:
                self.criterionPreLoss = VGGLoss_3D(
                    opt.pretrain_model_path,
                    device=self.device,
                ).to(self.device)

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr_max,
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.lr_max,
                betas=(opt.beta1, 0.999),
            )

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.mask = input['mask'].to(self.device)

    def forward(self, epoch, scaler=None):
        """
        One training step (D then G).
        If `scaler` is provided, uses GradScaler for mixed precision.
        Assumes outer code (Train.py) wraps this in torch.cuda.amp.autocast().
        """
        if not self.isTrain:
            with torch.no_grad():
                self.fake_B = self.netG(self.real_A)
            return

        use_amp = scaler is not None

        # ------------------ Generator forward ------------------
        self.fake_B = self.netG(self.real_A)

        # ------------------ D step ------------------
        set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad(set_to_none=True)

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach(), isDetach=True)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB, isDetach=True)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D_loss = (self.loss_D_fake + self.loss_D_real) * 0.5

        if use_amp:
            scaler.scale(self.loss_D_loss).backward()
            scaler.step(self.optimizer_D)
            scaler.update()
        else:
            self.loss_D_loss.backward()
            self.optimizer_D.step()

        # ------------------ G step ------------------
        set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad(set_to_none=True)

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB, isDetach=False)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1

        if self.VGG_loss:
            self.loss_G_perceive = self.criterionPreLoss(self.fake_B, self.real_B)
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceive
        else:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1

        if use_amp:
            scaler.scale(self.loss_G).backward()
            scaler.step(self.optimizer_G)
            scaler.update()
        else:
            self.loss_G.backward()
            self.optimizer_G.step()


def tensor2im(input_image, imtype=np.uint8):
    """
    Converts a Tensor array into a numpy image array.
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * 0.3081 + 0.1307) * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor2im3d(image_tensor):
    image_numpy = image_tensor[0].cpu().float().numpy()
    return image_numpy


def save_networks(opt, save_name, model, epoch):
    save_filename = '%s.pth' % (save_name)
    save_path = os.path.join(opt.model_results, save_filename)
    state = {
        'epoch': epoch + 1,
        'netG_state_dict': model.netG.state_dict(),
        'netD_state_dict': model.netD.state_dict() if hasattr(model, 'netD') else None,
        'optimizer_G': model.optimizer_G.state_dict() if hasattr(model, 'optimizer_G') else None,
        'optimizer_D': model.optimizer_D.state_dict() if hasattr(model, 'optimizer_D') else None,
    }
    torch.save(state, save_path)


def load_networks(opt, model):
    load_filename = f"{opt.load_name}.pth"
    load_path = os.path.join(opt.model_results, load_filename)

    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    print(f"Loading the model from {load_path}")

    # Always load to CPU first (safe for single-GPU & DDP)
    state = torch.load(load_path, map_location="cpu")

    # Load Generator
    netG_state = state.get("netG_state_dict", None)
    if netG_state is not None:
        model.netG.load_state_dict(netG_state, strict=True)
    else:
        raise KeyError("Checkpoint does not contain netG_state_dict")

    # Load Discriminator (train mode only)
    if opt.isTrain and hasattr(model, "netD"):
        netD_state = state.get("netD_state_dict", None)
        if netD_state is not None:
            model.netD.load_state_dict(netD_state, strict=True)

    # Load optimizers (ONLY when resuming training)
    if opt.isTrain and opt.continue_train:
        if hasattr(model, "optimizer_G") and state.get("optimizer_G") is not None:
            model.optimizer_G.load_state_dict(state["optimizer_G"])

        if hasattr(model, "optimizer_D") and state.get("optimizer_D") is not None:
            model.optimizer_D.load_state_dict(state["optimizer_D"])

    # Restore epoch counter
    opt.epoch_count = state.get("epoch", 1)

    print(f"Successfully loaded checkpoint '{opt.load_name}' ")


def print_current_message(epoch, iters, dataset_size, lr, iter_acc, losses):
    message = '(epoch: %d, iters: %d/%d, lr: %.6f, iter_acc: %.3f)' % (
        epoch,
        iters,
        dataset_size,
        lr,
        iter_acc,
    )
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)
    print(message)


def get_current_visuals(model):
    real_A = tensor2im3d(model.real_A.data)
    fake_B = tensor2im3d(model.fake_B.data)
    real_B = tensor2im3d(model.real_B.data)
    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])


def get_current_losses(model):
    errors_ret = OrderedDict()
    for name in model.loss_names:
        if isinstance(name, str):
            errors_ret[name] = float(getattr(model, 'loss_' + name))
    return errors_ret


def update_learning_rate(model, max_epochs, epoch, lr_max):
    model.optimizer_G.param_groups[0]['lr'] = lr_max * (1 - epoch / max_epochs) ** 0.9
    model.optimizer_D.param_groups[0]['lr'] = model.optimizer_G.param_groups[0]['lr']


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def inverser_norm_ct(x, max_vax, max_min):
    x = (x + 1) / 2
    x = (max_vax - max_min) * x - abs(max_min)
    return x


def print_options(opt):
    message = '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)
    if not opt.continue_train:
        with open(opt.file_name_txt, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')



def NiiDataRead(path, as_type=np.float32):
    """
    Read NIfTI (.nii / .nii.gz) and return:
      vol      : ndarray [z, y, x]
      spacing  : ndarray [z, y, x]  (float32)
      origin   : tuple
      direction: tuple
    """
    img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img).astype(as_type)  # z,y,x

    # SimpleITK spacing is (x, y, z); we store as (z, y, x)
    spacing = np.array(img.GetSpacing()[::-1], dtype=np.float32)
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return vol, spacing, origin, direction


def NiiDataWrite(save_path, vol, spacing, origin, direction, as_type=np.float32):
    """
    Write volume with proper spacing/origin/direction.
      vol     : ndarray [z, y, x]
      spacing : anything array-like of length 3 (z, y, x) (float-ish)
    """
    vol_np = np.asarray(vol, dtype=as_type)

    spacing_np = np.asarray(spacing, dtype=np.float64).reshape(-1)
    if spacing_np.size != 3:
        raise ValueError(f"Expected spacing of length 3 (z,y,x), got {spacing_np}")

    # SimpleITK expects spacing as (x, y, z)
    sitk_spacing = tuple(float(s) for s in spacing_np[::-1])

    img = sitk.GetImageFromArray(vol_np)
    img.SetSpacing(sitk_spacing)
    img.SetOrigin(tuple(origin))
    img.SetDirection(tuple(direction))
    sitk.WriteImage(img, save_path)


def N4BiasFieldCorrection(volumn_path, save_path):
    img = sitk.ReadImage(volumn_path)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    inputVolumn = sitk.Cast(img, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    sitk.WriteImage(corrector.Execute(inputVolumn, mask), save_path)


def dcm2nii(DCM_DIR, OUT_PATH):
    fuse_list = []
    for dicom_file in os.listdir(DCM_DIR):
        dicom = pydicom.dcmread(osp.join(DCM_DIR, dicom_file))
        fuse_list.append([dicom.pixel_array, float(dicom.SliceLocation)])
    fuse_list.sort(key=lambda x: x[1])
    volume_list = [i[0] for i in fuse_list]
    volume = np.array(volume_list).astype(np.float32) - 1024
    [spacing_x, spacing_y] = dicom.PixelSpacing
    spacing = np.array([dicom.SliceThickness, spacing_x, spacing_y])
    NiiDataWrite(OUT_PATH, volume, spacing, origin=(0, 0, 0), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))


def nii2mat(in_path, out_dir=None):
    volume, spacing, _, _ = NiiDataRead(in_path)
    if out_dir is None:
        out_path = in_path.split('.')[0] + '.mat'
    else:
        out_path = osp.join(out_dir, osp.split(in_path)[-1].split('.')[0] + '.mat')
    scipy.io.savemat(out_path, {'volume': volume, 'spacing': spacing})
    print(f'Saved at {out_path}')


def mat2nii(in_path, out_dir=None):
    mat_contents = scipy.io.loadmat(in_path)
    if out_dir is None:
        out_path = in_path.split('.')[0] + '.nii.gz'
    else:
        out_path = osp.join(out_dir, osp.split(in_path)[-1].split('.')[0] + '.nii.gz')
    NiiDataWrite(out_path, mat_contents['output'], mat_contents['spacing'][0], origin=(0, 0, 0),
                 direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))


def mha_read(path, as_type=np.float32):
    """
    Read .mha volume and return:
      vol      : ndarray [z, y, x]
      spacing  : ndarray [z, y, x]  (float32)
      origin   : tuple
      direction: tuple
    """
    img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img).astype(as_type)
    spacing = np.array(img.GetSpacing()[::-1], dtype=np.float32)
    origin = tuple(img.GetOrigin())
    direction = tuple(img.GetDirection())
    return vol, spacing, origin, direction


def mha_write(path, vol, spacing, origin, direction, as_type=np.float32):
    """
    Write MHA volume, preserving spacing/origin/direction
    (expects spacing in (z,y,x); SITK wants (x,y,z)).
    """
    vol_np = np.asarray(vol, dtype=as_type)
    spacing_np = np.asarray(spacing, dtype=np.float64).reshape(-1)
    if spacing_np.size != 3:
        raise ValueError(f"Expected spacing of length 3 (z,y,x), got {spacing_np}")
    sitk_spacing = tuple(float(s) for s in spacing_np[::-1])

    img = sitk.GetImageFromArray(vol_np)
    img.SetSpacing(sitk_spacing)
    img.SetOrigin(tuple(origin))
    img.SetDirection(tuple(direction))
    sitk.WriteImage(img, path)


def load_volume(path):
    if path.endswith(".nii.gz") or path.endswith(".nii"):
        return NiiDataRead(path)
    elif path.endswith(".mha"):
        return mha_read(path)
    else:
        raise ValueError(f"Unsupported image format: {path}")


def normalization(data, min_value, max_value):
    if not isinstance(data, np.ndarray):
        data = data.numpy()
    nor_data = (data - min_value) / (max_value - min_value)
    last_data = (nor_data - 0.5) * 2
    return last_data


def normalize_ct(ct, mask, ct_min, ct_max):
    ct = np.clip(ct, ct_min, ct_max)
    ct[mask == 0] = ct_min
    return normalization(ct, ct_min, ct_max).astype(np.float32)


SUPPORTED_EXT = ["nii.gz", "mha"]

def randomcrop_Npatch(crop_size, crop_Npatch, mri1, ct, ct_mask):
    this_frame = crop_size
    img = mri1
    non_zero_z, non_zero_x, non_zero_y = np.where(ct_mask == 1)
    non_zero_num = non_zero_x.shape[0]
    patch_index = random.sample(range(0, non_zero_num), crop_Npatch)

    patch_mri1 = np.zeros([crop_Npatch, *this_frame], dtype=np.float32)
    patch_ct   = np.zeros([crop_Npatch, *this_frame], dtype=np.float32)
    patch_mask = np.zeros([crop_Npatch, *this_frame], dtype=np.int32)

    D, H, W = this_frame
    half_D, half_H, half_W = D//2, H//2, W//2

    for idx in range(crop_Npatch):
        z_med = non_zero_z[patch_index[idx]]
        x_med = non_zero_x[patch_index[idx]]
        y_med = non_zero_y[patch_index[idx]]

        # Z
        if z_med < half_D:
            z0, z1 = 0, D
        elif z_med + half_D > img.shape[0]:
            z1 = img.shape[0]
            z0 = z1 - D
        else:
            z0, z1 = z_med - half_D, z_med + half_D

        # X
        if x_med < half_H:
            x0, x1 = 0, H
        elif x_med + half_H > img.shape[1]:
            x1 = img.shape[1]
            x0 = x1 - H
        else:
            x0, x1 = x_med - half_H, x_med + half_H

        # Y
        if y_med < half_W:
            y0, y1 = 0, W
        elif y_med + half_W > img.shape[2]:
            y1 = img.shape[2]
            y0 = y1 - W
        else:
            y0, y1 = y_med - half_W, y_med + half_W

        patch_mri1[idx] = mri1[z0:z1, x0:x1, y0:y1]
        patch_ct[idx]   = ct[z0:z1, x0:x1, y0:y1]
        patch_mask[idx] = ct_mask[z0:z1, x0:x1, y0:y1]

    return (np.ascontiguousarray(patch_mri1),
            np.ascontiguousarray(patch_ct),
            np.ascontiguousarray(patch_mask))


class DatasetFromFolder_train(Dataset):
    """
    DDP-friendly dataset using EXACT second-script patch extraction and
    synchronized horizontal flipping (MRI/CT/MASK).
    """

    def __init__(self, opt):
        self.root = opt.image_dir
        self.Max_CT = opt.Max_CT

        self.crop_size = [opt.depthSize, opt.ImageSize, opt.ImageSize]
        self.Npatch = opt.Npatch
        self.ran_num = 1  

        self.patient_dirs = sorted(glob.glob(os.path.join(self.root, "*")))
        if len(self.patient_dirs) == 0:
            raise RuntimeError(f"No patients found in {self.root}")

        print(f"[Dataset] Loading {len(self.patient_dirs)} patients...")

        self.data = []
        for folder in self.patient_dirs:
            pid = os.path.basename(folder)
            mr = self._load(folder, "mr")
            ct = self._load(folder, "ct")
            mask = self._load(folder, "mask")

            mr_norm = normalization(mr[0], 0, 255).astype(np.float32)

            ct_min = -1000
            ct_raw = np.clip(ct[0], ct_min, self.Max_CT)
            ct_raw[mask[0] == 0] = ct_min
            ct_norm = normalization(ct_raw, ct_min, self.Max_CT).astype(np.float32)

            self.data.append({
                "id": pid,
                "mr": mr_norm,
                "ct": ct_norm,
                "mask": mask[0].astype(np.float32),
                "spacing": mr[1],
                "origin": mr[2],
                "direction": mr[3],
            })

        self.total_patches = len(self.data) * self.Npatch
        print(f"[Dataset] Ready: {self.total_patches} total patches.")

    def _load(self, folder, name):
        for ext in SUPPORTED_EXT:
            path = os.path.join(folder, f"{name}.{ext}")
            if os.path.exists(path):
                return load_volume(path)
        raise FileNotFoundError(f"{name} not found in folder {folder}")

    def __getitem__(self, idx):
        vol_idx = idx // self.Npatch
        sample  = self.data[vol_idx]

        mr, ct, mask = sample["mr"], sample["ct"], sample["mask"]

        A_np, B_np, M_np = randomcrop_Npatch(
            self.crop_size, self.ran_num, mr, ct, mask
        )

        A = torch.from_numpy(A_np)  
        B = torch.from_numpy(B_np)
        M = torch.from_numpy(M_np)


        if random.random() < 0.5:
            A = torch.flip(A, dims=[3])  
            B = torch.flip(B, dims=[3])
            M = torch.flip(M, dims=[3])

        return {
            "A": A,  # MRI
            "B": B,  # CT
            "mask": M,
            "patient_id": sample["id"],
            "spacing": sample["spacing"],
            "origin": sample["origin"],
            "direction": sample["direction"],
        }

    # ----------------------------------------------------------------------

    def __len__(self):
        return self.total_patches
