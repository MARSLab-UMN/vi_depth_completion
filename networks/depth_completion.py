import logging
from typing import List
import torch
import torch.nn as nn
import torchvision
import collections

from networks import network_utils



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResNetPyramids(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, resnet_arch=101):
        super(ResNetPyramids, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(resnet_arch)](pretrained=pretrained)

        self.channel = in_channels

        self.conv1 = nn.Sequential(collections.OrderedDict([
            ('conv1_1', nn.Conv2d(self.channel, 64, kernel_size=3, stride=2, padding=1, bias=False)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn_2', nn.BatchNorm2d(64)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1_3', nn.BatchNorm2d(128)),
            ('relu1_3', nn.ReLU(inplace=True))
        ]))
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer1[0].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        if pretrained:
            network_utils.weights_init(self.conv1, 'kaiming')
            network_utils.weights_init(self.layer1[0].conv1, 'kaiming')
            network_utils.weights_init(self.layer1[0].downsample[0], 'kaiming')
        else:
            network_utils.weights_init(self.modules(), 'kaiming')


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


class ModifiedFPN(nn.Module):
    def __init__(self):
        super(ModifiedFPN, self).__init__()
        self.resnet_rgb = ResNetPyramids(in_channels=3, pretrained=True)
        self.resnet_normal = ResNetPyramids(in_channels=3, pretrained=False)
        self.resnet_depth = ResNetPyramids(in_channels=1, pretrained=False)

        self.feature1_upsamping = nn.Sequential(
            nn.Conv2d(256 * 3, 256 * 3, 3, 1, 1),
            nn.BatchNorm2d(256 * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 * 3, 128 * 3, 1),
            nn.BatchNorm2d(128 * 3),
            nn.ReLU(inplace=True),
        )
        self.feature2_upsamping = nn.Sequential(
            nn.Conv2d(512 * 3, 256 * 3, 1),
            nn.BatchNorm2d(256 * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 * 3, 256 * 3, 3, 1, 1),
            nn.BatchNorm2d(256 * 3),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256 * 3, 128 * 3, 1),
            nn.BatchNorm2d(128 * 3),
            nn.ReLU(inplace=True),
        )
        self.feature3_upsamping = nn.Sequential(
            nn.Conv2d(1024 * 3, 512 * 3, 1),
            nn.BatchNorm2d(512 * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 * 3, 512 * 3, 3, 1, 1),
            nn.BatchNorm2d(512 * 3),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512 * 3, 256 * 3, 1),
            nn.BatchNorm2d(256 * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 * 3, 256 * 3, 3, 1, 1),
            nn.BatchNorm2d(256 * 3),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256 * 3, 128 * 3, 1),
            nn.BatchNorm2d(128 * 3),
            nn.ReLU(inplace=True),
        )
        self.feature4_upsamping = nn.Sequential(
            nn.Conv2d(2048 * 3, 1024 * 3, 1),
            nn.BatchNorm2d(1024 * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024 * 3, 1024 * 3, 3, 1, 1),
            nn.BatchNorm2d(1024 * 3),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(15, 20)),
            nn.Conv2d(1024 * 3, 512 * 3, 1),
            nn.BatchNorm2d(512 * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 * 3, 512 * 3, 3, 1, 1),
            nn.BatchNorm2d(512 * 3),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512 * 3, 256 * 3, 1),
            nn.BatchNorm2d(256 * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 * 3, 256 * 3, 3, 1, 1),
            nn.BatchNorm2d(256 * 3),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256 * 3, 128 * 3, 1),
            nn.BatchNorm2d(128 * 3),
            nn.ReLU(inplace=True),
        )

        self.feature_concat = nn.Sequential(
            nn.Conv2d(128 * 3, 64 * 3, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 * 3, 1, 1, 1, 1),
            nn.UpsamplingBilinear2d(size=(240, 320)),
            nn.ReLU(inplace=True),
        )

        logging.info('ModifiedFPN number of parameters: {0}.'.format(count_parameters(self)))

    def combine_rgbdn(self, rgb, normal, depth, level):
        return torch.cat((rgb, normal, depth), dim=1)

    def forward(self, image: torch.Tensor, normal: torch.Tensor, incomplete_depth: torch.Tensor):
        i1, i2, i3, i4 = self.resnet_rgb(image)
        n1, n2, n3, n4 = self.resnet_normal(normal)
        d1, d2, d3, d4 = self.resnet_depth(incomplete_depth)

        z1 = self.feature1_upsamping(self.combine_rgbdn(i1, n1, d1, 0))
        z2 = self.feature2_upsamping(self.combine_rgbdn(i2, n2, d2, 1))
        z3 = self.feature3_upsamping(self.combine_rgbdn(i3, n3, d3, 2))
        z4 = self.feature4_upsamping(self.combine_rgbdn(i4, n4, d4, 3))

        y = self.feature_concat(z1 + z2 + z3 + z4)
        return y
