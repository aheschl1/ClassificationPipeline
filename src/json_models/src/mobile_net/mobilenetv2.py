from typing import Dict

import torch.nn as nn
from src.json_models.src.modules import PolyWrapper, XModule, PXModule


class DWSeperable(nn.Module):
    def __init__(self, in_channels, out_channels, stride, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            # depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # point
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )


def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )


class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride, conv, conv_args: Dict):
        super(InvertedBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = ch_in * expand_ratio
        self.use_res_connect = self.stride == 1 and ch_in == ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))

        layers.extend([
            conv(hidden_dim, ch_out, stride=stride, **conv_args),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6()
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)


class MobileNetV2(nn.Module):
    def __init__(self, ch_in=3, conv: str = 'DW', conv_args: Dict = None):
        super(MobileNetV2, self).__init__()
        assert conv in ['DW', 'Conv', 'Poly', 'XModule', 'PXModule']
        if conv_args is None:
            conv_args = {}
        self.configs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        conv_op = None
        if conv == 'DW':
            conv_op = DWSeperable
        elif conv == 'Conv':
            conv_op = nn.Conv2d
        elif conv == 'Poly':
            conv_op = PolyWrapper
        elif conv == 'XModule':
            conv_op = XModule
        elif conv == 'PXModule':
            conv_op = PXModule

        self.stem_conv = conv3x3(ch_in, 32, stride=2)

        layers = []
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, 
                                            stride=stride, conv=conv_op, conv_args=conv_args))
                input_channel = c

        self.layers = nn.Sequential(*layers)

        self.last_conv = conv1x1(input_channel, 1280)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 1000)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).view(-1, 1280)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # model check
    model = MobileNetV2(ch_in=3, n_classes=1000)
