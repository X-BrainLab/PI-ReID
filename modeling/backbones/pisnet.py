# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
from torch import nn
from time import time
from torch.nn import functional as F

from modeling.backbones.Query_Guided_Attention import Query_Guided_Attention
import numpy as np

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def feature_corruption(x_g, x_g2):
    # We ABANDON the standard feature corruption in the paper.
    # The simple concat yields the comparable performance.
    corrupted_x = torch.cat((x_g, x_g2), 3)
    return corrupted_x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class pisnet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3], has_non_local="no", sia_reg="no", pyramid="no"):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)
        print("has_non_local:" + has_non_local)
        self.has_non_local = has_non_local
        self.pyramid = pyramid
        self.Query_Guided_Attention = Query_Guided_Attention(in_channels=2048)
        self.Query_Guided_Attention.apply(weights_init_kaiming)
        self.sia_reg = sia_reg


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_g, x, x_g2=[], is_first=False):


        x = self.conv1(x)
        x_g = self.conv1(x_g)

        x = self.bn1(x)
        x_g = self.bn1(x_g)

        x = self.maxpool(x)
        x_g = self.maxpool(x_g)

        x = self.layer1(x)
        x_g = self.layer1(x_g)

        x = self.layer2(x)
        x_g = self.layer2(x_g)

        x = self.layer3(x)
        x_g = self.layer3(x_g)

        x = self.layer4(x)
        x_g = self.layer4(x_g)

        if not isinstance(x_g2, list):

            x_g2 = self.conv1(x_g2)
            x_g2 = self.bn1(x_g2)
            x_g2 = self.maxpool(x_g2)
            x_g2 = self.layer1(x_g2)
            x_g2 = self.layer2(x_g2)
            x_g2 = self.layer3(x_g2)
            x_g2 = self.layer4(x_g2)

        x1, attention1 = self.Query_Guided_Attention(x, x_g, attention='x', pyramid=self.pyramid)

        if not isinstance(x_g2, list):
            x2, attention2 = self.Query_Guided_Attention(x, x_g2, attention='x', pyramid=self.pyramid)
            if self.sia_reg == "yes":
                rec_x_g = feature_corruption(x_g, x_g2.detach())
                x3, attention3 = self.Query_Guided_Attention(x1, rec_x_g, attention='x_g', pyramid=self.pyramid)
        else:
            x2 = []
            attention2 = []
            x3 = []
            attention3 = []

        if isinstance(is_first, tuple):
            x1[0, :, :, :] = x_g[0, :, :, :]

        return x1, attention1, x2, attention2, x_g, x3, attention3

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()