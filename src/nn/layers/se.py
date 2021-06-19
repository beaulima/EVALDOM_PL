import torch
import typing
import thelper
from thelper.nn.activations import get_activation_layer

# https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf

class SqueezeExcitationLayer(torch.nn.Module):

    def __init__(self, channel, reduction=16, activation="relu"):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction),
            get_activation_layer(activation),
            torch.nn.Linear(channel // reduction, channel),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SqueezeExcitationBlock(torch.nn.Module):

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, reduction=16, coordconv=False,
                 radius_channel=True, activation="relu"):
        super().__init__()
        self.conv1 = thelper.nn.layers.coordconv.make_conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.activ = get_activation_layer(activation)
        self.conv2 = self._make_conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.se = SqueezeExcitationLayer(planes, reduction, activation=activation)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activ(out)
        return out


class ModifiedSqueezeExcitationBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride=1,
                 downsample=None, reduction=16, coordconv=False,
                 radius_channel=True, activation="swish"):
        super().__init__()
        self.conv1 = thelper.nn.layers.coordconv.make_conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.activ = get_activation_layer(activation)
        self.conv2 = thelper.nn.layers.coordconv.make_conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.se = SqueezeExcitationLayer(out_channels, reduction, activation=activation)
        self.downsample = downsample

    def forward(self, x):

        out = self.conv1(x)
        out = self.activ(out)
        residual = out
        out = self.conv2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activ(out)
        return (out - torch.mean(out)) / torch.std(out)
