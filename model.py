import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable

class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        x = self.body(x)
        return x
        
class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class make_dense_ca(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense_ca, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
        self.CA = CALayer(growthRate)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = self.CA(out)
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB_CA(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RDB_CA, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense_ca(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out
class RDN_CA(nn.Module):
    def __init__(self, args):
        super(RDN_CA, self).__init__()
        nChannel = args.nChannel
        nDenselayer = args.nDenselayer
        nBlock = args.nBlock
        nFeat = args.nFeat
        scale = args.scale
        growthRate = args.growthRate
        self.args = args

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs 3

        self.RDB1 = RDB_CA(nFeat, nDenselayer, growthRate)
        '''
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        '''
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*nBlock, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat*scale*scale, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale)
        # conv
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1, bias=True)
        #self.CAT = torch.cat(feature, 1)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(self.args.nFeat, self.args.nDenselayer, self.args.growthRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        #print(x.data.numpy().shape)
        F_  = self.conv1(x)
        #print(F_.data.numpy().shape)
        F_0 = self.conv2(F_)
        #print(F_0.data.numpy().shape)
        '''
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        '''
        feature = []
        for i in range(self.args.nBlock):
            F_0 = self.RDB1(F_0)
            feature.append(F_0)
            #print(F_0.data.numpy().shape)
            #print(FF.data.numpy().shape)
        FF = torch.cat(feature, 1)
        #print(FF.data.numpy().shape)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        us = self.conv_up(FDF)
        us = self.upsample(us)

        #output = self.conv3(us)


        return us
# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out
# Residual Dense Network
class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        nChannel = args.nChannel
        nDenselayer = args.nDenselayer
        nBlock = args.nBlock
        nFeat = args.nFeat
        scale = args.scale
        growthRate = args.growthRate
        self.args = args

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs 3

        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        '''
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        '''
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*nBlock, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat*scale*scale, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale)
        # conv 
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1, bias=True)
        #self.CAT = torch.cat(feature, 1)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(self.args.nFeat, self.args.nDenselayer, self.args.growthRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        #print(x.data.numpy().shape)
        F_  = self.conv1(x)
        #print(F_.data.numpy().shape)
        F_0 = self.conv2(F_)
        #print(F_0.data.numpy().shape)
        '''
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        '''
        feature = []
        for i in range(self.args.nBlock):
            F_0 = self.RDB1(F_0)
            feature.append(F_0) 
            #print(F_0.data.numpy().shape)
            #print(FF.data.numpy().shape)
        FF = torch.cat(feature, 1)                
        #print(FF.data.numpy().shape)                
        FdLF = self.GFF_1x1(FF)         
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        us = self.conv_up(FDF)
        us = self.upsample(us)

        #output = self.conv3(us)


        return us

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # np.transpose(x,(2,3,0,1)) = torch.transpose(torch.transpose(x,0,2),1,3)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.bn1(self.conv1(F.pad(x, (1, 1, 1, 1), "replicate"))))
        # print("conv1 output for conv2: ")
        # print output.shape
        output = self.bn2(self.conv2(F.pad(output, (1, 1, 1, 1), "replicate")))
        output = torch.add(output, identity_data)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn_mid = nn.BatchNorm2d(64)

        # self.upscale4x = nn.Sequential(
        #    nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
        #    nn.PixelShuffle(2),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
        #    nn.PixelShuffle(2),
        #    nn.LeakyReLU(0.2, inplace=True),
        # )
        self.upscale4x_a = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upscale4x_b = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=6, kernel_size=9, stride=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(F.pad(x, (4, 4, 4, 4), "replicate")))
        # print("out for conv_mid: ")
        # print out.shape
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(F.pad(out, (1, 1, 1, 1), "replicate")))
        # print("out for upscale4x: ")
        # print out.shape
        out = torch.add(out, residual)
        # out = self.upscale4x_11(F.pad(out,(1,1,1,1),"replicate"))
        out = self.upscale4x_a(F.pad(out, (1, 1, 1, 1), "replicate"))
        out = self.upscale4x_b(F.pad(out, (1, 1, 1, 1), "replicate"))
        # print("out for conv_output: ")
        # print out.shape
        #out = self.conv_output(F.pad(out, (4, 4, 4, 4), "replicate"))
        return out

class final_layer(nn.Module):
    def __init__(self, out_channels):
        super(final_layer, self).__init__()

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=9, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.conv_output(F.pad(x, (4, 4, 4, 4), "replicate"))
        return out

class final_layer1(nn.Module):
    def __init__(self, out_channels):
        super(final_layer1, self).__init__()

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = self.conv_output(x)
        return out
