import torch
from torch import nn
from torch.nn import functional as F

from plane_mask_detection.maskrcnn_benchmark.modeling.make_layers import group_norm
from plane_mask_detection.maskrcnn_benchmark.modeling.make_layers import make_fc
from plane_mask_detection.maskrcnn_benchmark.modeling.make_layers import make_conv3x3

import math # to get pi

def init_weights(m):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class UpConvNet(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(UpConvNet, self).__init__()
        # TODO: Make these as input parameters
        in_channels = 256
        out_channels = 256
        self.cfg = cfg

        # highest resolution
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.Upsample(size=(64, 80),  mode='bilinear', align_corners=True),
                        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),

                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.Upsample(size=(64, 80), mode='bilinear', align_corners=True),
                        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
                        )

        self.conv4 = nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),

                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),

                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.Upsample(size=(64, 80), mode='bilinear', align_corners=True),
                        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                        )

        self.conv5 = nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),

                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),

                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),

                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.Upsample(size=(64, 80), mode='bilinear', align_corners=True),
                        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                        )
        # 0: estimate directly 3 values of normal vector w/o any particular constraints, follows by normalization, then L1 loss
        # 1: estimate 2dof [tt, phi] only, then L1 loss
        # 2: estimate 3 value of normal, applying tanh in the last layer (scale up to -1 .. 1), no normalization needed, then MSE loss
        # 3: estimate 3 value of normal, applying hard-tanh in the last layer (scale up to -1 .. 1), no normalization needed, then MSE loss
        # 4: estimate 2dof, followed by a periodic characteristic, then L1 loss
        if self.cfg.SOLVER.OUTPUT_REPRESENTATION == 0 or self.cfg.SOLVER.OUTPUT_REPRESENTATION == 6:
            self.conv_output = nn.Sequential(
                                nn.Conv2d(out_channels, 3, kernel_size=1, stride=1, padding=0),
                                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        elif self.cfg.SOLVER.OUTPUT_REPRESENTATION == 1:
            self.conv_output_tt = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
                # nn.Tanh(),
                nn.Upsample(scale_factor=4, mode='bilinear'))
            self.conv_output_phi = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
                nn.Tanh(),
                nn.Upsample(scale_factor=4, mode='bilinear'))
        elif self.cfg.SOLVER.OUTPUT_REPRESENTATION == 2:
            self.conv_output = nn.Sequential(
                nn.Conv2d(out_channels, 3, kernel_size=1, stride=1, padding=0),
                nn.Tanh(),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        elif self.cfg.SOLVER.OUTPUT_REPRESENTATION == 3:
            self.conv_output = nn.Sequential(
                nn.Conv2d(out_channels, 3, kernel_size=1, stride=1, padding=0),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                nn.BatchNorm2d(3), nn.Hardtanh())
        elif self.cfg.SOLVER.OUTPUT_REPRESENTATION == 4:
            self.conv_output_tt = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
                nn.Tanh(),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
            self.conv_output_phi = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
                nn.Tanh(),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        elif self.cfg.SOLVER.OUTPUT_REPRESENTATION == 5:
            self.conv_output_sttctt = nn.Sequential(
                nn.Conv2d(out_channels, 2, kernel_size=1, stride=1, padding=0),
                # nn.Hardtanh(),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
            self.conv_output_phi = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
                nn.Hardtanh(),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        '''
        elif self.cfg.SOLVER.OUTPUT_REPRESENTATION == 5:
            self.conv_output_tt = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                nn.Tanh())
            self.conv_output_phi = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                nn.Tanh())
        '''

    def forward(self, features):
        x1 = self.conv1(features[0])
        x2 = self.conv2(features[1])
        x3 = self.conv3(features[2])
        x4 = self.conv4(features[3])
        x5 = self.conv5(features[4])
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)

        ## NOTE: Due to size_divisibility = 32 while the max resize factor is 64, here we need this code to interpolate the last
        ## Upsampling convolution layer to the correct size
        ## Also Note that the input is expected in the format of:  mini-batch x channels x [optional depth] x [optional height] x width
        if self.cfg.SOLVER.OUTPUT_REPRESENTATION == 1 or self.cfg.SOLVER.OUTPUT_REPRESENTATION == 4:
            y = torch.cat( (self.conv_output_tt(x1 + x2 + x3 + x4 + x5),
                            self.conv_output_phi(x1 + x2 + x3 + x4 + x5)),
                           dim = 1)
        elif self.cfg.SOLVER.OUTPUT_REPRESENTATION == 5:
            y = torch.cat((self.conv_output_sttctt(x1 + x2 + x3 + x4 + x5),
                           self.conv_output_phi(x1 + x2 + x3 + x4 + x5)),
                          dim=1)
        else:
            y = self.conv_output(x1+x2+x3+x4+x5)
        return y
