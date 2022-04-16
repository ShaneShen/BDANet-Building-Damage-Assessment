import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models
#import apex
import apex.parallel.sync_batchnorm as BN
from .senet import se_resnext50_32x4d, senet154
#from .dpn import dpn92
from torch.nn.parameter import Parameter
from thop import profile
from .antialias import Downsample as downsamp

class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualDownSample, self).__init__()
        self.bot = nn.Sequential(downsamp(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        out = self.bot(x)
        return out

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualDownSample(in_channels))
            in_channels = int(in_channels * stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        out = self.bot(x)

        return out

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualUpSample(in_channels))
            in_channels = int(in_channels // stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class ConvReluBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvReluBN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class ConvBNReluNkernel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, BN = True):
        super(ConvBNReluNkernel, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size == 5:
            self.conv= nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = self.kernel_size, padding=2)
        if kernel_size == 3:
            self.conv= nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = self.kernel_size, padding=1)
        if kernel_size == 1:
            self.conv= nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = self.kernel_size, padding=0)
        if BN == True:
            self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.BN(self.conv(x)))
        return y


class Attention_block(nn.Module):
    def __init__(self, F_c, F_de,  reduction=16, concat=True):
        super(Attention_block,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(F_c, F_c//reduction, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 =  nn.Conv2d(F_c//reduction, F_de, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()


        self.spatial_se = nn.Sequential(nn.Conv2d(F_de, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())
        self.concat = concat

    def forward(self,f, x):
        f=self.avg_pool(f)
        f = self.fc1(f)
        f = self.relu(f)
        f = self.fc2(f)
        chn_se = self.sigmoid(f)
        chn_se = chn_se * x

        spa_se = self.spatial_se(x)
        spa_se = x * spa_se

        if self.concat:
            return torch.cat([chn_se, spa_se], dim=1)
        else:
            return chn_se + spa_se


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True, )
        )
    #@autocast()
    def forward(self, x):
        return self.layer(x)

class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=16):
        super(BasicResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
        # self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        # self.norm = nn.GroupNorm(num_groups=16, num_channels=out_channels)
        # self.relu= nn.ReLU(inplace=True)
    #@autocast()
    def forward(self, x):
        x = self.layer(x)
        # x = self.conv3(x)
        # x = self.norm(x)
        #
        # x = self.relu(x)
        return x

class SCSEModule2(nn.Module):
    # according to https://arxiv.org/pdf/1808.08127.pdf concat is better
    def __init__(self, channels, reduction=16, concat=False):
        super(SCSEModule2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

        self.spatial_se = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())
        self.concat = concat

    def forward(self, x):
        module_input = x

        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        chn_se = self.sigmoid(x)
        chn_se = chn_se * module_input + module_input

        spa_se = self.spatial_se(module_input)
        #chn_se = chn_se * spa_se
        spa_se = module_input * spa_se + module_input
        if self.concat:
            return torch.cat([chn_se, spa_se], dim=1)
        else:
            return (chn_se + spa_se)/2


class SCSEModule(nn.Module):
    # according to https://arxiv.org/pdf/1808.08127.pdf concat is better
    def __init__(self, channels, reduction=16, concat=False):
        super(SCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

        self.spatial_se = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())
        self.concat = concat

    def forward(self, x):
        module_input = x

        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        chn_se = self.sigmoid(x)
        chn_se = chn_se * module_input

        spa_se = self.spatial_se(module_input)
        spa_se = module_input * spa_se
        if self.concat:
            return torch.cat([chn_se, spa_se], dim=1)
        else:
            return chn_se + spa_se



class SeResNext50_Unet_MScale(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(SeResNext50_Unet_MScale, self).__init__()

        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = [48,  128, 256, 256, 2048]
        f_filter = 24

        # self.convF2 = ConvRelu(encoder_filters[1]+encoder_filters[1], encoder_filters[1])
        # self.convF3 = ConvRelu(encoder_filters[2]+encoder_filters[1], encoder_filters[2])
        # self.convF4 = ConvRelu(encoder_filters[3]+encoder_filters[1], encoder_filters[3])

        # self.convF1 = nn.Sequential(ConvRelu(f_filter + encoder_filters[0], encoder_filters[0]), BasicResBlock(encoder_filters[0], encoder_filters[0]))
        # self.convF2 = nn.Sequential(ConvRelu(encoder_filters[1]+ encoder_filters[1], encoder_filters[1]), BasicResBlock(encoder_filters[1], encoder_filters[1]))
        # self.convF3 = nn.Sequential(ConvRelu(encoder_filters[1]+ encoder_filters[2], encoder_filters[2]), BasicResBlock(encoder_filters[2], encoder_filters[2]))
        self.convF1 = nn.Sequential( BasicResBlock(f_filter + encoder_filters[0], encoder_filters[0]))
        self.convF2 = nn.Sequential( BasicResBlock(f_filter + encoder_filters[1], encoder_filters[1]))
        #self.convF3 = nn.Sequential(BasicResBlock(f_filter +encoder_filters[2], encoder_filters[2]))
        #self.convF4 = nn.Sequential(ConvRelu(encoder_filters[1]+ encoder_filters[3], encoder_filters[3]), BasicResBlock(encoder_filters[3], encoder_filters[3]))
        self.xconv256 =  nn.Sequential(BasicResBlock(3, f_filter))
        self.xconv128 =  nn.Sequential(BasicResBlock(3, f_filter))
        #self.xconv64 = nn.Sequential(BasicResBlock(3, f_filter))
        #self.xconv32 =  nn.Sequential(ConvReluBN(in_channels=3, out_channels=64), encoder.layer1 )
        self.dconv5 = ConvRelu(encoder_filters[4], decoder_filters[3])
        self.dconv5_2 = ConvRelu(encoder_filters[3]+decoder_filters[3], decoder_filters[3])

        self.dconv6 = ConvRelu(decoder_filters[3], decoder_filters[2])
        self.dconv6_2 = ConvRelu(encoder_filters[2]+decoder_filters[2], decoder_filters[2])
        self.dconv7 = ConvRelu(decoder_filters[2], decoder_filters[1])
        self.dconv7_2 = ConvRelu(encoder_filters[1]+decoder_filters[1], decoder_filters[1])
        self.dconv8 = ConvRelu(decoder_filters[1], decoder_filters[0])
        self.dconv8_2 = ConvRelu(encoder_filters[0]+decoder_filters[0], decoder_filters[0])

        self.dconv9 = ConvRelu(decoder_filters[0], decoder_filters[0])

        self.res = nn.Conv2d(decoder_filters[0]*2, 5, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = se_resnext50_32x4d(pretrained=pretrained)
        #encoder = torchvision.models.resnet50(pretrained=pretrained)
        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1) #encoder.layer0.conv1
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

        #self.xconv128 =  nn.Sequential(ConvReluBN(in_channels=3, out_channels=64), encoder.layer1 )
        #self.xconv64 = nn.Sequential(ConvReluBN(in_channels=3, out_channels=64), encoder.layer1 )
        #self.xconv32 =  nn.Sequential(ConvReluBN(in_channels=3, out_channels=64), encoder.layer1 )

    def forward1(self, x):
        batch_size, C, H, W = x.shape
        x256 = F.interpolate(x, scale_factor=0.5)
        x128 = F.interpolate(x, scale_factor=0.25)
        #x64 = F.interpolate(x, scale_factor=0.125)
        #x32 = F.interpolate(x, scale_factor=0.0625)

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        enc256 = self.xconv256(x256)
        enc128 = self.xconv128(x128)
        #enc64 = self.xconv64(x64)
        #enc32 = self.xconv32(x32)

        enc1 = self.convF1(torch.cat([enc1, enc256],1))
        enc2 = self.convF2(torch.cat([enc2, enc128], 1))
        #enc3 = self.convF3(torch.cat([enc3, enc64], 1))
        #enc4 = self.convF4(torch.cat([enc4, enc32], 1))


        dec5 = self.dconv5(F.interpolate(enc5, scale_factor=2))
        dec5 = self.dconv5_2(torch.cat([dec5,enc4], 1))
        dec6 = self.dconv6(F.interpolate(dec5, scale_factor=2))
        dec6 = self.dconv6_2(torch.cat([dec6,enc3], 1))
        dec7 = self.dconv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.dconv7_2(torch.cat([dec7,enc2], 1))
        dec8 = self.dconv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.dconv8_2(torch.cat([dec8,enc1], 1))

        dec9 = self.dconv9(F.interpolate(dec8,scale_factor=2))

        return dec9

    def forward(self, x):
        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])

        dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SeResNext50_Unet_MScale2(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(SeResNext50_Unet_MScale2, self).__init__()

        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = [48,  128, 256, 256, 2048]
        f_filter = 24

        # self.convF2 = ConvRelu(encoder_filters[1]+encoder_filters[1], encoder_filters[1])
        # self.convF3 = ConvRelu(encoder_filters[2]+encoder_filters[1], encoder_filters[2])
        # self.convF4 = ConvRelu(encoder_filters[3]+encoder_filters[1], encoder_filters[3])

        # self.convF1 = nn.Sequential(ConvRelu(f_filter + encoder_filters[0], encoder_filters[0]), BasicResBlock(encoder_filters[0], encoder_filters[0]))
        # self.convF2 = nn.Sequential(ConvRelu(encoder_filters[1]+ encoder_filters[1], encoder_filters[1]), BasicResBlock(encoder_filters[1], encoder_filters[1]))
        # self.convF3 = nn.Sequential(ConvRelu(encoder_filters[1]+ encoder_filters[2], encoder_filters[2]), BasicResBlock(encoder_filters[2], encoder_filters[2]))
        #self.convF1 = nn.Sequential( BasicResBlock(f_filter + encoder_filters[0], encoder_filters[0]))
        self.convF2 = nn.Sequential( BasicResBlock(f_filter + encoder_filters[1], encoder_filters[1]))
        #self.convF3 = nn.Sequential(BasicResBlock(f_filter +encoder_filters[2], encoder_filters[2]))
        #self.convF4 = nn.Sequential(ConvRelu(encoder_filters[1]+ encoder_filters[3], encoder_filters[3]), BasicResBlock(encoder_filters[3], encoder_filters[3]))
        #self.xconv256 =  nn.Sequential(BasicResBlock(3, f_filter))
        self.xconv128 =  nn.Sequential(BasicResBlock(3, f_filter))
        #self.xconv64 = nn.Sequential(BasicResBlock(3, f_filter))
        #self.xconv32 =  nn.Sequential(ConvReluBN(in_channels=3, out_channels=64), encoder.layer1 )
        #self.dconv5 = ConvRelu(encoder_filters[4], decoder_filters[3])
        #self.dconv5_2 = ConvRelu(encoder_filters[3]+decoder_filters[3], decoder_filters[3])

        self.dconv6 = ConvReluBN(encoder_filters[3], decoder_filters[2])
        self.dconv6_2 = ConvRelu(encoder_filters[2]+decoder_filters[2], decoder_filters[2])
        self.dconv7 = ConvRelu(decoder_filters[2], decoder_filters[1])
        self.dconv7_2 = ConvRelu(encoder_filters[1]+decoder_filters[1], decoder_filters[1])
        self.dconv8 = ConvRelu(decoder_filters[1], decoder_filters[0])
        self.dconv8_2 = ConvRelu(encoder_filters[0]+decoder_filters[0], decoder_filters[0])
        self.dconv8_2 = nn.Sequential( ConvRelu(encoder_filters[0]+decoder_filters[0], decoder_filters[0]), SCSEModule(decoder_filters[0], reduction=2, concat=True))

        self.dconv9 = ConvRelu(decoder_filters[0]*2, decoder_filters[0])

        self.res = nn.Conv2d(decoder_filters[0]*2, 5, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = se_resnext50_32x4d(pretrained=pretrained)
        #encoder = torchvision.models.resnet50(pretrained=pretrained)
        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1) #encoder.layer0.conv1
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        #self.conv5 = encoder.layer4

        #self.xconv128 =  nn.Sequential(ConvReluBN(in_channels=3, out_channels=64), encoder.layer1 )
        #self.xconv64 = nn.Sequential(ConvReluBN(in_channels=3, out_channels=64), encoder.layer1 )
        #self.xconv32 =  nn.Sequential(ConvReluBN(in_channels=3, out_channels=64), encoder.layer1 )

    def forward1(self, x):
        batch_size, C, H, W = x.shape
        #x256 = F.interpolate(x, scale_factor=0.5)
        x128 = F.interpolate(x, scale_factor=0.25)
        #x64 = F.interpolate(x, scale_factor=0.125)
        #x32 = F.interpolate(x, scale_factor=0.0625)

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        #enc5 = self.conv5(enc4)

        #enc256 = self.xconv256(x256)
        enc128 = self.xconv128(x128)
        #enc64 = self.xconv64(x64)
        #enc32 = self.xconv32(x32)

        #enc1 = self.convF1(torch.cat([enc1, enc256],1))
        enc2 = self.convF2(torch.cat([enc2, enc128], 1))
        #enc3 = self.convF3(torch.cat([enc3, enc64], 1))
        #enc4 = self.convF4(torch.cat([enc4, enc32], 1))


        #dec5 = self.dconv5(F.interpolate(enc5, scale_factor=2))
        #dec5 = self.dconv5_2(torch.cat([dec5,enc4], 1))
        dec6 = self.dconv6(F.interpolate(enc4, scale_factor=2))
        dec6 = self.dconv6_2(torch.cat([dec6,enc3], 1))
        dec7 = self.dconv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.dconv7_2(torch.cat([dec7,enc2], 1))
        dec8 = self.dconv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.dconv8_2(torch.cat([dec8,enc1], 1))

        dec9 = self.dconv9(F.interpolate(dec8,scale_factor=2))

        return dec9

    def forward(self, x):
        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])

        dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SeResNext50_Unet_Double(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(SeResNext50_Unet_Double, self).__init__()

        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])


        self.res = nn.Conv2d(decoder_filters[-5] * 2, 5, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = se_resnext50_32x4d(pretrained=pretrained)
        #encoder = torchvision.models.resnet50(pretrained=pretrained)

        # conv1_new = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # _w = encoder.layer0.conv1.state_dict()
        # _w['weight'] = torch.cat([0.5 * _w['weight'], 0.5 * _w['weight']], 1)
        # conv1_new.load_state_dict(_w)
        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1) #encoder.layer0.conv1
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4


    def forward1(self, x):
        batch_size, C, H, W = x.shape

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4 ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3 ], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2 ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9,  enc1  ], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10


    def forward(self, x):
        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])

        dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SeResNext50_Unet_MultiScale(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(SeResNext50_Unet_MultiScale, self).__init__()

        #encoder_filters = [64, 256, 512, 1024, 2048]
        encoder_filters = [128, 256, 512, 1024, 2048]
        fuse_filter = 64

        self.down12 = DownSample(encoder_filters[0], 2)
        self.down13 = DownSample(encoder_filters[0]*2, 2)
        self.down23 = DownSample(encoder_filters[1], 2)
        self.up21 = UpSample(encoder_filters[1], 2)
        self.up31 = UpSample(encoder_filters[2]//2, 2)
        self.up32 = UpSample(encoder_filters[2], 2)

        # self.convF1 = ConvBNReluNkernel(encoder_filters[0], decoder_filters[0])
        # self.convF2 = ConvBNReluNkernel(encoder_filters[1], decoder_filters[1])
        # self.convF3 = ConvBNReluNkernel(encoder_filters[2], decoder_filters[2])
        self.conv0 = ConvRelu(encoder_filters[0]//2, encoder_filters[0])
        self.convF1 = nn.Sequential(ConvRelu(encoder_filters[0], encoder_filters[0]), SCSEModule(encoder_filters[0], reduction=4, concat=True))
        self.conv1_1 = ConvRelu(encoder_filters[0]*2, encoder_filters[0])
        self.convF2 = nn.Sequential(ConvRelu(encoder_filters[1], encoder_filters[1]), SCSEModule(encoder_filters[1], reduction=8, concat=True))
        self.conv2_1 = ConvRelu(encoder_filters[1]*2, encoder_filters[1])
        self.convF3 = nn.Sequential(ConvRelu(encoder_filters[2], encoder_filters[2]), SCSEModule(encoder_filters[2], reduction=16, concat=True))
        self.conv3_1 = ConvRelu(encoder_filters[2]*2, encoder_filters[2])

        self.conv1_2 = ConvRelu(encoder_filters[0] * 2, fuse_filter)
        self.conv2_2 = ConvRelu(encoder_filters[1] * 2, fuse_filter)
        self.conv3_2 = ConvRelu(encoder_filters[2] * 2, fuse_filter*2)
        self.up31_2 = UpSample(fuse_filter*2, 4)   #32
        self.up21_2 = UpSample(fuse_filter, 2)   #32

        self.conv4 = ConvRelu(fuse_filter * 2, fuse_filter)  # 32+32+64
        self.conv4_2 = nn.Conv2d(fuse_filter, fuse_filter, 1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        self.res = nn.Conv2d(fuse_filter*2, 5, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = se_resnext50_32x4d(pretrained=pretrained)
        #encoder = torchvision.models.resnet50(pretrained=pretrained)
        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1) #encoder.layer0.conv1
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        #self.conv4 = encoder.layer3


    def forward1(self, x):
        batch_size, C, H, W = x.shape

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)

        enc1 = self.conv0(enc1)

        f12 = self.down12(enc1)
        f13 = self.down13(f12)
        f23 = self.down23(enc2)
        f21 = self.up21(enc2)
        f32 = self.up32(enc3)
        f31 = self.up31(f32)

        fusion1 = self.convF1(enc1+f21+f31)
        fusion1 = self.conv1_1(fusion1)
        fusion2 = self.convF2(enc2+f12+f32)
        fusion2 = self.conv2_1(fusion2)
        fusion3 = self.convF3(enc3+f23+f13)
        fusion3 = self.conv3_1(fusion3)

        dec1 = self.conv1_2(torch.cat([enc1, fusion1], 1))
        dec2 = self.conv2_2(torch.cat([enc2, fusion2], 1))
        dec3 = self.conv3_2(torch.cat([enc3, fusion3], 1))

        dec2 = self.up21_2(dec2)
        dec3 = self.up31_2(dec3)
        dec4 = self.conv4(torch.cat([dec1, dec2, dec3], 1))
        dec4 = self.conv4_2(F.interpolate(dec4, scale_factor=2, mode='bilinear'))
        dec4 = self.relu(dec4)

        return dec4

    def forward(self, x):
        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])

        dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SeResNext50_Unet_2SUnet(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(SeResNext50_Unet_2SUnet, self).__init__()

        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3] , decoder_filters[-2] )
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4] , decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        # self.conv9_3 = nn.Sequential(ConvRelu(encoder_filters[-4], encoder_filters[-4]), nn.Sigmoid())
        # self.convx9_3 = ConvRelu(encoder_filters[-4], encoder_filters[-4])

        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        self.convxx = nn.Sequential(ConvRelu(decoder_filters[-5]*2, decoder_filters[-5]*2),
                                    nn.Conv2d(decoder_filters[-5] * 2, decoder_filters[-5], 1, stride=1, padding=0))


        self.res = nn.Conv2d(decoder_filters[-5] * 2, 5, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = se_resnext50_32x4d(pretrained=pretrained)
        #encoder = torchvision.models.resnet50(pretrained=pretrained)
        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1) #encoder.layer0.conv1
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4


    def forward1(self, x):
        batch_size, C, H, W = x.shape
        xx = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        encx1 = self.conv1(xx)   # 64 128 128
        encx2 = self.conv2(encx1) # 64
        encx3 = self.conv3(encx2) # 32
        encx4 = self.conv4(encx3) # 16
        encx5 = self.conv5(encx4) # 8

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4 ], 1))
        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9,  enc1], 1))   #256


        decx6 = self.conv6(F.interpolate(encx5, scale_factor=2))
        decx6 = self.conv6_2(torch.cat([decx6, encx4 ], 1))
        decx7 = self.conv7(F.interpolate(decx6, scale_factor=2))
        decx7 = self.conv7_2(torch.cat([decx7, encx3], 1))
        decx8 = self.conv8(F.interpolate(decx7, scale_factor=2))
        decx8 = self.conv8_2(torch.cat([decx8, encx2], 1))
        decx9 = self.conv9(F.interpolate(decx8, scale_factor=2))
        decx9 = self.conv9_2(torch.cat([decx9,  encx1], 1))   #128
        #decx9 = self.convx9_3(F.interpolate(decx9, scale_factor=4))


        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        decx10 = self.conv10(F.interpolate(decx9, scale_factor=4))

        dec = self.convxx(torch.cat([dec10, decx10], 1))

        return dec

    def forward(self, x):
        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])

        dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SeResNext50_Unet_2Ssum(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(SeResNext50_Unet_2Ssum, self).__init__()

        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3] , decoder_filters[-2] )
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4] , decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])

        # self.convx9_3 = ConvRelu(encoder_filters[-4], encoder_filters[-4])


        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        self.conv10_s = nn.Sequential(ConvRelu(decoder_filters[-5], decoder_filters[-5]),
                                      nn.Conv2d(decoder_filters[-5] , 1, 1, stride=1, padding=0),
                                      nn.Sigmoid())
        # self.convxx = nn.Sequential(ConvRelu(decoder_filters[-5]*2, decoder_filters[-5]*2),
        #                             nn.Conv2d(decoder_filters[-5] * 2, decoder_filters[-5], 1, stride=1, padding=0))

        self.res = nn.Conv2d(decoder_filters[-5] * 2, 5, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = se_resnext50_32x4d(pretrained=pretrained)
        #encoder = torchvision.models.resnet50(pretrained=pretrained)
        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1) #encoder.layer0.conv1
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4


    def forward1(self, x):
        batch_size, C, H, W = x.shape
        xx = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        encx1 = self.conv1(xx)   # 64 128 128
        encx2 = self.conv2(encx1) # 64
        encx3 = self.conv3(encx2) # 32
        encx4 = self.conv4(encx3) # 16
        encx5 = self.conv5(encx4) # 8

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4 ], 1))
        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9,  enc1], 1))   #256

        decx6 = self.conv6(F.interpolate(encx5, scale_factor=2))
        decx6 = self.conv6_2(torch.cat([decx6, encx4 ], 1))
        decx7 = self.conv7(F.interpolate(decx6, scale_factor=2))
        decx7 = self.conv7_2(torch.cat([decx7, encx3], 1))
        decx8 = self.conv8(F.interpolate(decx7, scale_factor=2))
        decx8 = self.conv8_2(torch.cat([decx8, encx2], 1))
        decx9 = self.conv9(F.interpolate(decx8, scale_factor=2))
        decx9 = self.conv9_2(torch.cat([decx9,  encx1], 1))   #128
        #decx9 = self.convx9_3(F.interpolate(decx9, scale_factor=4))


        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        alpha = self.conv10_s(dec10)
        decx10 = self.conv10(F.interpolate(decx9, scale_factor=4))

        dec = alpha * dec10 + (1-alpha)*decx10

        return dec

    def forward(self, x):
        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])

        dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == '__main__':
    import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    size_input = 512 #817 #577 #473 #713
    input = torch.rand(1, 6, size_input, size_input)#.cuda()
    model = SeResNext50_Unet_MultiScale()#.cuda()
    #macs, params = profile(model, inputs=(input, ))
    #print(macs, params)
    model.eval()
    print(model)
    output = model(input)
    print('PyConvSegNet', output.size())
