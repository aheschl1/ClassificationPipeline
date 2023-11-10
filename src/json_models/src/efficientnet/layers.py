import torch
from torch import nn


class ConvBnAct(nn.Module):
    """Layer grouping a convolution, batchnorm, and activation function"""

    def __init__(self, n_in, n_out, conv_op, conv_args=None, kernel_size=3,
                 stride=1, padding=0, groups=1, bias=False,
                 bn=True, act=True):
        super().__init__()
        if conv_args is None:
            conv_args = {}

        self.conv = conv_op(n_in, n_out, kernel_size=kernel_size,
                            stride=stride, padding=padding,
                            groups=groups, bias=bias, **conv_args)
        self.bn = nn.BatchNorm2d(n_out) if bn else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-excitation block"""

    def __init__(self, n_in, r=24):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Conv2d(n_in, n_in // r, kernel_size=1),
                                        nn.SiLU(),
                                        nn.Conv2d(n_in // r, n_in // r, kernel_size=1),
                                        nn.Sigmoid())

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        print(y.shape, x.shape)
        return x * y


class DropSample(nn.Module):
    """Drops each sample in x with probability p during training"""

    def __init__(self, p=0):
        super().__init__()

        self.p = p

    def forward(self, x):
        if (not self.p) or (not self.training):
            return x

        batch_size = x.shape[0]
        random_tensor = torch.FloatTensor(batch_size, 1, 1, 1).uniform_().to(x.device)
        bit_mask = self.p < random_tensor

        x = x.div(1 - self.p)
        x = x * bit_mask
        return x


class MBConvN(nn.Module):
    """MBConv with an expansion factor of N, plus squeeze-and-excitation"""

    def __init__(self, n_in, n_out, expansion_factor, conv_op, conv_args: dict, kernel_size=3, stride=1, r=4, p=0):
        super().__init__()

        padding = (kernel_size - 1) // 2
        expanded = expansion_factor * n_in
        self.skip_connection = (n_in == n_out) and (stride == 1)

        self.expand_pw = nn.Identity() if (expansion_factor == 1) else ConvBnAct(n_in, expanded, conv_op,
                                                                                 conv_args=conv_args, kernel_size=1)
        self.depthwise = ConvBnAct(expanded, expanded, conv_op, conv_args=conv_args, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=expanded)
        self.se = SEBlock(expanded, r=r * expansion_factor)
        self.reduce_pw = ConvBnAct(expanded, n_out, conv_op, conv_args=conv_args, kernel_size=1,
                                   act=False)
        self.dropsample = DropSample(p)

    def forward(self, x):
        residual = x

        x = self.expand_pw(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.reduce_pw(x)

        if self.skip_connection:
            x = self.dropsample(x)
            x = x + residual

        return x


class MBConv1(MBConvN):
    def __init__(self, n_in, n_out, conv_op, conv_args, kernel_size=3,
                 stride=1, r=24, p=0):
        super().__init__(n_in, n_out, 1, conv_op, conv_args,
                         kernel_size=kernel_size, stride=stride,
                         r=r, p=p)


class MBConv6(MBConvN):
    def __init__(self, n_in, n_out, conv_op, conv_args, kernel_size=3,
                 stride=1, r=24, p=0):
        super().__init__(n_in, n_out, expansion_factor=6,
                         conv_op=conv_op, conv_args=conv_args,
                         kernel_size=kernel_size, stride=stride,
                         r=r, p=p)
