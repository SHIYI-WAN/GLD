import torch
import torch.nn as nn
from GConv import GDConv
import torch.utils.checkpoint as cp
import matplotlib.pyplot as plt

def gabdconv7x7(in_channels, M, n_scale, stride=1, expand=False):
    return GDConv(in_channels, in_channels, kernel_size=5, M=M, nScale=n_scale, stride=stride,
                    padding=2, dilation=1, groups=in_channels, bias=False, expand=expand, padding_mode='zeros')

def gabor_block_function_factory(conv, gconv, norm, relu=None):
    def block_function(x):
        if conv is not None:
            x = conv(x)
        x = gconv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x
    return block_function

def do_efficient_fwd(block_f,x,efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block_f,x)
    else:
        return block_f(x)


class GaborEncoderBlock_v2(nn.Module):
    def __init__(self, in_c, out_c, orientation, scale, conv1, expand = False, downsample = None, efficient=True, use_bn=True):
        super(GaborEncoderBlock_v2, self).__init__()

        self.efficient = efficient
        if conv1:
            self.conv1x1 = nn.Conv2d(in_c, out_c*orientation, kernel_size=1, bias=False)
        else:
            self.conv1x1 = None
        if expand:
            self.conv_scale1 = gabdconv7x7(in_c, orientation, scale[0], stride=1, expand=True)
        else:
            self.conv_scale1 = gabdconv7x7(out_c, orientation, scale[0], stride=1)
        self.bn1 = nn.BatchNorm2d(out_c*orientation) if use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv_scale2 = gabdconv7x7(out_c, orientation, scale[1], stride=2)
        self.bn2 = nn.BatchNorm2d(out_c*orientation) if use_bn else None


    def forward(self, x):
        block_f1 = gabor_block_function_factory(self.conv1x1, self.conv_scale1, self.bn1, self.relu)
        block_f2 = gabor_block_function_factory(None, self.conv_scale2,self.bn2)

        out = do_efficient_fwd(block_f1,x,self.efficient)
        out = do_efficient_fwd(block_f2,out,self.efficient)
        relu_out = self.relu(out)

        return relu_out, relu_out



class GaborEncoder_v2(nn.Module):
    def __init__(self, input_channels=8, channels=[32,48,80]):
        super(GaborEncoder_v2, self).__init__()
        orientation = 4
        scale = [1, 2]
        g_channels = [int(i/orientation) for i in channels]

        self.block1 = GaborEncoderBlock_v2(input_channels, g_channels[0], orientation, scale, conv1=False, expand=True)
        self.block2 = GaborEncoderBlock_v2(channels[0], g_channels[1], orientation, scale, conv1=True)
        # self.block3 = GaborEncoderBlock_v2(channels[1], g_channels[2], orientation, scale, conv1=True)
        #self.block4 = GaborEncoderBlock_v2(channels[2], g_channels[3], orientation, scale, conv1=True)

    def forward(self, x):
        features = []
        x, skip1 = self.block1(x)
        features += [skip1]
        x, skip2 = self.block2(x)
        features += [skip2]
        # x, skip3 = self.block3(x)
        # features += [skip3]
        # x, skip = self.block4(x)
        # features += [skip]
        return skip1,skip2#,skip3

# How to call the encoder
def gabor_encoder_v2():
    input_channels=8#8
    channels = [32, 48, 80]
    encoder = GaborEncoder_v2(input_channels=input_channels, channels=channels)

    return encoder



