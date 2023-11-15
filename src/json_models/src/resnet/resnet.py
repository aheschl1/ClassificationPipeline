import torch.nn as nn

ADD = 'add'
CONCAT = 'concat'


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, mode=ADD):
        pass
