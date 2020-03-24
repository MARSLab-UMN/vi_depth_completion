import torch
from torch import nn
from torch.nn import functional as F

from plane_mask_detection.maskrcnn_benchmark.modeling.make_layers import group_norm
from plane_mask_detection.maskrcnn_benchmark.modeling.make_layers import make_fc
from plane_mask_detection.maskrcnn_benchmark.modeling.make_layers import make_conv3x3

import math # to get pi

class ResGravityUpConvNet(nn.Module):
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
        super(ResGravityUpConvNet, self).__init__()
        # TODO: Make these as input parameters
        in_channels = 256
        out_channels = 256
        self.cfg = cfg

        # highest resolution
        self.upsamp_feat0 = nn.Upsample(size=(64, 80),  mode='bilinear', align_corners=True)
        self.upsamp_feat1 = nn.Upsample(size=(32, 40), mode='bilinear', align_corners=True)
        self.upsamp_feat2 = nn.Upsample(size=(16, 20), mode='bilinear', align_corners=True)
        self.upsamp_feat3 = nn.Upsample(size=(8, 10), mode='bilinear', align_corners=True)
        self.upsamp_feat4 = nn.Upsample(size=(4, 5), mode='bilinear', align_corners=True)

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

        if self.cfg.SOLVER.OUTPUT_REPRESENTATION == 0 or self.cfg.SOLVER.OUTPUT_REPRESENTATION == 6:
            self.conv_output = nn.Sequential(
                                nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0),
                                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        elif self.cfg.SOLVER.OUTPUT_REPRESENTATION == 1:
            self.conv_output_tt = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
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

    def forward(self, features, gravity_dir):
        z0 = features[0][:, 0:3, :, :] + self.upsamp_feat0(gravity_dir)
        x1 = self.conv1(torch.cat((z0, features[0][:, 3:, :, :]), dim=1))
        z1 = features[1][:, 0:3, :, :] + self.upsamp_feat1(gravity_dir)
        x2 = self.conv2(torch.cat((z1, features[1][:, 3:, :, :]), dim=1))
        z2 = features[2][:, 0:3, :, :] + self.upsamp_feat2(gravity_dir)
        x3 = self.conv2(torch.cat((z2, features[2][:, 3:, :, :]), dim=1))
        z3 = features[3][:, 0:3, :, :] + self.upsamp_feat3(gravity_dir)
        x4 = self.conv4(torch.cat((z3, features[3][:, 3:, :, :]), dim=1))
        z4 = features[4][:, 0:3, :, :] + self.upsamp_feat4(gravity_dir)
        x5 = self.conv5(torch.cat((z4, features[4][:, 3:, :, :]), dim=1))

        if self.cfg.SOLVER.OUTPUT_REPRESENTATION == 1 or self.cfg.SOLVER.OUTPUT_REPRESENTATION == 4:
            y = torch.cat( (self.conv_output_tt(x1 + x2 + x3 + x4 + x5),
                            self.conv_output_phi(x1 + x2 + x3 + x4 + x5)),
                            dim = 1)
        elif self.cfg.SOLVER.OUTPUT_REPRESENTATION == 5:
            y = torch.cat((self.conv_output_sttctt(x1 + x2 + x3 + x4 + x5),
                           self.conv_output_phi(x1 + x2 + x3 + x4 + x5)),
                           dim=1)
        else:
            x6 = x1 + x2 + x3 + x4 + x5
            z6 = x6[:, 0:3, :, :] + self.upsamp_feat0(gravity_dir)
            y = self.conv_output(torch.cat((x6, z6[:, 3:, :, :]), dim=1))
        return y
