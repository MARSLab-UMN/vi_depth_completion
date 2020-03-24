import torch
import torch.nn as nn


def weights_init(modules, init_type='xavier'):
    assert init_type == 'xavier' or init_type == 'kaiming'
    m = modules
    if (isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or \
            isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.ConvTranspose2d) or
            isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Linear)):
        if init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    #elif isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
        #for m in modules:
            #weights_init(m, init_type)
    elif isinstance(m, nn.Module):
        for _, m in modules.named_children():
            weights_init(m, init_type)
