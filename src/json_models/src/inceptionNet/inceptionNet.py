import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict
import torch.nn as nn
from src.json_models.src.modules import PolyWrapper, XModule, PXModule, MultiRoute, MultiBatchNorm
from src.json_models.src.mobile_net.mobilenetv2 import DWSeperable

class ConvBlock(nn.Module,):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm2d(out_chanels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ConvBlockv2(nn.Module):

    def __init__(self, in_channels, out_chanels, conv = "Conv",conv_args=None, **kwargs):
        super(ConvBlockv2, self).__init__()
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
        elif conv == 'MultiRoute':
            conv_op = MultiRoute
        #print(conv_args['order'],conv_op)
        if conv_args==None:
            conv_args=dict()
        conv_args.update(kwargs)
        if conv=='Conv':
            self.conv = conv_op(in_channels, out_chanels,**kwargs)
        elif conv=='XModule':
            kernel_sizes = [kwargs['kernel_size']]
            print(kernel_sizes)
            self.conv = conv_op(in_channels, out_chanels, kernel_sizes=kernel_sizes)
        else:
            self.conv = conv_op(in_channels, out_chanels, **conv_args)
        self.bn = nn.BatchNorm2d(out_chanels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
    
    
class InceptionBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_1x1,
        red_3x3,
        out_3x3,
        red_5x5,
        out_5x5,
        out_pool,
        conv = "conv",conv_args=None
    ):
        super(InceptionBlock, self).__init__()
        self.conv = conv
        #print(conv)
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            ConvBlockv2(red_3x3, out_3x3, kernel_size=3, padding=1, conv = conv, conv_args=conv_args),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlockv2(red_5x5, out_5x5, kernel_size=5, padding=2, conv = conv,conv_args=conv_args),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)


class InceptionV1(nn.Module):
    def __init__(self, aux_logits=False, num_classes=1000, conv='conv', conv_args = None):
        super(InceptionV1, self).__init__()
        assert conv in ['DW', 'Conv', 'Poly', 'XModule', 'PXModule', 'MultiRoute']

        if conv_args is None:
            self.conv_args = {}
        self.aux_logits = aux_logits
        self.conv1 = ConvBlock(
            in_channels=3, 
            out_chanels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32,conv,conv_args)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64, conv,conv_args)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64,conv,conv_args)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64,conv,conv_args)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64, conv,conv_args)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64,conv,conv_args)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64,conv,conv_args)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64, conv,conv_args)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128, conv,conv_args)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128, conv,conv_args)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128, conv,conv_args)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)
        


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)
        
        
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        
        x = self.inception4e(x)
        x = self.maxpool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x