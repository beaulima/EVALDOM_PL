import math

import torch
import torch.nn
import torch.nn as nn
from src.nn.layers.coordconv import CoordConv2d, CoordConvTranspose2d, AddCoords, make_conv2d
from src.nn.layers.srm import setup_srm_layer, setup_srm_weights
from src.nn.layers.lambda_layer import LambdaLayer

warned_bad_input_size_power2 = False

from src.utils import utils
log = utils.get_logger(__name__)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

class BasicBlock(torch.nn.Module):
    """Default (double-conv) block used in U-Net layers."""

    def __init__(self, in_channels, out_channels, coordconv=False, kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coordconv = coordconv
        self.kernel_size = kernel_size
        self.padding = padding
        self.layer = torch.nn.Sequential(
            make_conv2d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, padding=padding, coordconv=coordconv),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),

            make_conv2d(in_channels=out_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, padding=padding, coordconv=coordconv),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class BasicBlockLambda(torch.nn.Module):
    """Default (double-conv) block used in U-Net layers."""

    def __init__(self, in_channels, out_channels, coordconv=False, kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coordconv = coordconv
        self.kernel_size = kernel_size
        self.padding = padding
        self.layer = torch.nn.Sequential(
            make_conv2d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, padding=padding, coordconv=coordconv),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),

            LambdaLayer(out_channels),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class BasicBlockNoBn(torch.nn.Module):
    """Modified (double-conv) block used in U-Net layers."""

    def __init__(self, in_channels, out_channels, coordconv=False, kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coordconv = coordconv
        self.kernel_size = kernel_size
        self.padding = padding

        self.phantom_bn1 = torch.nn.BatchNorm2d(out_channels)
        self.phantom_bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv1 = make_conv2d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, padding=padding, coordconv=coordconv)

        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = make_conv2d(in_channels=out_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, padding=padding, coordconv=coordconv)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        residual = out
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return (out - torch.mean(out))/torch.std(out)


class AttentionBlock(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class UpConv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_channels, out_channels, upscale="ConvTranspose"):
        super(UpConv, self).__init__()
        if upscale == "ConvTranspose":
            self.up = torch.nn.ConvTranspose2d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=2, stride=2)
        elif upscale == "Upsample":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif upscale == "Unpool":
            self.up = nn.Sequential(
                nn.MaxUnpool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(torch.nn.Module):
    """U-Net implementation. Not identical to the original.

    This version includes batchnorm and transposed conv2d layers for upsampling. Coordinate Convolutions
    (CoordConv) can also be toggled on if requested (see :mod:`thelper.nn.layers.coordconv` for more information).
    """

    def __init__(self, in_channels=3, mid_channels=512, num_classes=2, coordconv=False, srm=False, upscale="ConvTranspose",
                 basic_block_func="default"):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.coordconv = coordconv
        self.srm = srm
        self.pool = torch.nn.MaxPool2d(2)
        self.basic_block_func = None
        if basic_block_func == "default":
            self.basic_block_func = BasicBlock
        elif basic_block_func == "modified_no_bn":
            self.basic_block_func = BasicBlockNoBn

        #self.basic_block_func = thelper.nn.layers.se.ModifiedSqueezeExcitationBlock

        self.srm_conv = srm.setup_srm_layer(in_channels) if srm else None
        self.encoder_block1 = self.basic_block_func(in_channels=in_channels + 3 if srm else in_channels,
                                         out_channels=mid_channels // 16,
                                         coordconv=coordconv)
        self.encoder_block2 = self.basic_block_func(in_channels=mid_channels // 16,
                                         out_channels=mid_channels // 8,
                                         coordconv=coordconv)
        self.encoder_block3 = self.basic_block_func(in_channels=mid_channels // 8,
                                         out_channels=mid_channels // 4,
                                         coordconv=coordconv)
        self.encoder_block4 = self.basic_block_func(in_channels=mid_channels // 4,
                                         out_channels=mid_channels // 2,
                                         coordconv=coordconv)
        self.mid_block = self.basic_block_func(in_channels=mid_channels // 2,
                                    out_channels=mid_channels,
                                    coordconv=coordconv)

        self.upsampling_block4 = UpConv(in_channels=mid_channels,  out_channels=mid_channels // 2, upscale=upscale)
        self.decoder_block4 = self.basic_block_func(in_channels=mid_channels,
                                         out_channels=mid_channels // 2,
                                         coordconv=coordconv)
        self.upsampling_block3= UpConv(in_channels=mid_channels // 2, out_channels=mid_channels // 4, upscale=upscale)
        self.decoder_block3 = self.basic_block_func(in_channels=mid_channels // 2,
                                         out_channels=mid_channels // 4,
                                         coordconv=coordconv)
        self.upsampling_block2 = UpConv(in_channels=mid_channels // 4, out_channels=mid_channels // 8, upscale=upscale)
        self.decoder_block2 = self.basic_block_func(in_channels=mid_channels // 4,
                                         out_channels=mid_channels // 8,
                                         coordconv=coordconv)
        self.upsampling_block1 = UpConv(in_channels=mid_channels // 8, out_channels=mid_channels // 16, upscale=upscale)
        self.num_classes = num_classes

        self.final_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.mid_channels // 8,
                            out_channels=self.mid_channels // 16,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=self.mid_channels // 16,
                            out_channels=self.num_classes,
                            kernel_size=1),
        )

        if not coordconv:
            self.weight_init()

    def weight_init(self):
        for m in self.modules():
            weights_init_kaiming(m)

    def forward(self, x):
        global warned_bad_input_size_power2
        if not warned_bad_input_size_power2 and len(x.shape) == 4:
            if not math.log(x.shape[-1], 2).is_integer() or not math.log(x.shape[-2], 2).is_integer():
                warned_bad_input_size_power2 = True
                log.warning("unet input size should be power of 2 (e.g. 256x256, 512x512, ...)")
        if self.srm_conv is not None:
            noise = self.srm_conv(x)
            x = torch.cat([x, noise], dim=1)
        encoded1 = self.encoder_block1(x)  # 512x512
        encoded2 = self.encoder_block2(self.pool(encoded1))  # 256x256
        encoded3 = self.encoder_block3(self.pool(encoded2))  # 128x128
        encoded4 = self.encoder_block4(self.pool(encoded3))  # 64x64
        # encoder 5
        embedding = self.mid_block(self.pool(encoded4))  # 32x32
        decoded4 = self.decoder_block4(torch.cat([encoded4, self.upsampling_block4(embedding)], dim=1))
        decoded3 = self.decoder_block3(torch.cat([encoded3, self.upsampling_block3(decoded4)], dim=1))
        decoded2 = self.decoder_block2(torch.cat([encoded2, self.upsampling_block2(decoded3)], dim=1))
        # decoder1
        out = self.final_block(torch.cat([encoded1, self.upsampling_block1(decoded2)], dim=1))
        return out
