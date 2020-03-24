import torch
from torch import nn
from torch.nn import functional as F

from plane_mask_detection.maskrcnn_benchmark.modeling.make_layers import group_norm
from plane_mask_detection.maskrcnn_benchmark.modeling.make_layers import make_fc
from plane_mask_detection.maskrcnn_benchmark.modeling.make_layers import make_conv3x3

import math # to get pi

def init_weights(modules):
    print('WEIGHT INITIALIZATION: \n', modules)
    m = modules
    if isinstance(m, nn.Conv2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            m.weight.data.fill_(1.0)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Module):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    m.weight.data.fill_(1.0)

                if m.bias is not None:
                    m.bias.data.zero_()

class FullImageEncoder(nn.Module):
    def __init__(self):
        super(FullImageEncoder, self).__init__()
        self.dropout = nn.Dropout2d(p=0.5)
        self.global_fc = nn.Linear(1024 * 8 * 10, 512)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1
        self.upsample = nn.Upsample(size=(120, 160), mode='nearest') # just mean copy the same neuron to large spatial
        init_weights(self.modules())

    def forward(self, x):
        # x: (H=480)/64 x (W=640)/64 x 1024 = 8x10x1024
        x1 = self.dropout(x)
        x2 = x1.view(-1, 1024 * 8 * 10)
        x3 = self.relu(self.global_fc(x2))
        x3 = x3.view(-1, 512, 1, 1)
        x4 = self.conv1(x3)
        out = self.upsample(x4) # 512x120x160
        return out

class DORN(nn.Module):
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
        super(DORN, self).__init__()
        # TODO: Make these as input parameters
        in_channels = 1024
        out_channels = 512
        self.cfg = cfg
        self.full_image_encode = FullImageEncoder()

        # SKIP highest resolution
        # self.conv1 = nn.Sequential(
        #                 nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        #                 nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True),
                        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),

                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
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
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
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
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                        )

        # 0: estimate directly 3 values of normal vector w/o any particular constraints, follows by normalization, then L1 loss
        # 1: estimate 2dof [tt, phi] only, then L1 loss
        # 2: estimate 3 value of normal, applying tanh in the last layer (scale up to -1 .. 1), no normalization needed, then MSE loss
        # 3: estimate 3 value of normal, applying hard-tanh in the last layer (scale up to -1 .. 1), no normalization needed, then MSE loss
        # 4: estimate 2dof, followed by a periodic characteristic, then L1 loss
        if self.cfg.SOLVER.OUTPUT_REPRESENTATION == 0:
            self.conv_output = nn.Sequential(
                                                nn.Dropout2d(p=0.5),
                                                nn.Conv2d(512 * 5, 2048, 1),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout2d(p=0.5),
                                                nn.Conv2d(2048, 3, 1), # 3 channels output
                                                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
                                            )
        else:
            print('SOLVER OUTPUT_REPRESENTATION NOT IMPLEMENTED YET!')
            exit(0)
        # elif self.cfg.SOLVER.OUTPUT_REPRESENTATION == 1:
        #     self.conv_output_tt = nn.Sequential(
        #         nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
        #         nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        #         nn.Tanh())
        #     self.conv_output_phi = nn.Sequential(
        #         nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
        #         nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        #         nn.Tanh())
        # elif self.cfg.SOLVER.OUTPUT_REPRESENTATION == 2:
        #     self.conv_output = nn.Sequential(
        #         nn.Conv2d(out_channels, 3, kernel_size=1, stride=1, padding=0),
        #         nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        #         nn.Tanh())
        # elif self.cfg.SOLVER.OUTPUT_REPRESENTATION == 3:
        #     self.conv_output = nn.Sequential(
        #         nn.Conv2d(out_channels, 3, kernel_size=1, stride=1, padding=0),
        #         nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        #         nn.BatchNorm2d(3), nn.Hardtanh())
        # elif self.cfg.SOLVER.OUTPUT_REPRESENTATION == 4:
        #     self.conv_output_tt = nn.Sequential(
        #         nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
        #         nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        #         nn.Hardtanh(min_val=-math.pi, max_val=math.pi))
        #     self.conv_output_phi = nn.Sequential(
        #         nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
        #         nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), nn.Hardtanh(min_val=0, max_val=math.pi/2.0))

    def forward(self, features):
        x1 = self.conv1(features[0])
        x2 = self.conv2(features[1])
        x3 = self.conv3(features[2])
        x4 = self.conv4(features[3])
        x5 = self.conv5(features[4])

        # DONE 1. Change the upsample size to a fixed dimension for x1 - x5
        # DONE 2. Add Full image Encoder
        # 3. Replace the sum by concatenation followed by a 2D dropout, this dropout will scale the information from
        # multiple channels properly, preventing overfitting
        # 4. Let the final output as 2D conv w/o any modification
        # 5. Increase the input/output dimension (both in the MaskRCNN and in here)
        # 6. Train w/ normal only with cosine embedding loss
        print('dorn.py: x1 shape: ', x1.shape)
        print('dorn.py: x2 shape: ', x2.shape)
        print('dorn.py: x3 shape: ', x3.shape)
        print('dorn.py: x4 shape: ', x4.shape)
        print('dorn.py: x5 shape: ', x5.shape)
        x5 = torch.nn.functional.interpolate(x5, size=(x4.shape[2], x4.shape[3]),
                                                       mode='bilinear', align_corners=True)

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        print('dorn.py: x5 shape: ', x5.shape)
        print('dorn.py: x6 shape: ', x6.shape)

        if self.cfg.SOLVER.OUTPUT_REPRESENTATION == 1 or self.cfg.SOLVER.OUTPUT_REPRESENTATION == 4:
            y = torch.cat( (self.conv_output_tt(x1 + x2 + x3 + x4 + x5),
                            self.conv_output_phi(x1 + x2 + x3 + x4 + x5)),
                           dim = 1)
        else:
            y = self.conv_output(x6)

        print('dorn.py: y shape: ', y.shape)
        return y
