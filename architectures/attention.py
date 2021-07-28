import torch
from torch import nn
from collections import OrderedDict
from .base import concat, get_activation, conv, conv2dbn
from .mulresunet import Block2d


class ChannelGate(nn.Module):
    """
        The channel block. process the feature extracted by encoder and decoder.
        (Convolutional block attention module)
        referenc (squeeze-and-excitation network)
    """
    
    def __init__(self, f_x, reduction_ratio=4):
        super(ChannelGate, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.psi = nn.Sequential(
            nn.Conv2d(f_x, f_x // reduction_ratio, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(f_x // reduction_ratio, f_x, kernel_size=1, stride=1, padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_max = self.psi(self.maxpool(x))
        x_avg = self.psi(self.avgpool(x))
        return x * self.sigmoid(x_max + x_avg)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    """
    The channel block. process the feature extracted by encoder and decoder.
    """
    
    def __init__(self, f_x, kernel_size=7):
        super(SpatialGate, self).__init__()
        kernel_size = kernel_size
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        return x * x_out


class CBAM(nn.Module):
    """
    The convolutional block attention module
    """
    
    def __init__(self, f_x, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(f_x, reduction_ratio)
        self.SpatialGate = SpatialGate(f_x, kernel_size=kernel_size)
    
    def forward(self, x):
        return self.SpatialGate(self.ChannelGate(x))


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


def attention(f_x, kind='unet', reduce_ratio=8, kernel_size=7):
    if kind == 'cbam':
        return CBAM(f_x, reduction_ratio=reduce_ratio, kernel_size=kernel_size)
    else:
        return Identity()


class GridAttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(GridAttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            conv(F_g, F_int, 1, 1),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            conv(F_l, F_int, 3, 2),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            conv(F_int, 1, 1, 1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class AttentionUnet(nn.Module):
    def __init__(self, fin=3,
                 fout=1,
                 act_fun='LeakyReLU',
                 need_bias=True,
                 att='cbam',
                 reduce_ratio=4):
        super(AttentionUnet, self).__init__()
        self.downblock1 = nn.Sequential(OrderedDict({
            'conv1': conv2dbn(fin, 16, 3, 1, need_bias, act_fun),
            'conv2': conv2dbn(16, 16, 3, 1, need_bias, act_fun),
            'att1' : attention(16, kind=att, reduce_ratio=reduce_ratio, kernel_size=7)
        }))  # B X 16 X H X W
        self.downblock2 = nn.Sequential(OrderedDict({
            'downconv': nn.MaxPool2d(2, 2),
            'conv1'   : conv2dbn(16, 32, 3, 1, need_bias, act_fun),
            'conv2'   : conv2dbn(32, 32, 3, 1, need_bias, act_fun),
            'att2'    : attention(32, kind=att, reduce_ratio=reduce_ratio, kernel_size=7)
        }))  # B X 32 X H/2 X W/2
        self.downblock3 = nn.Sequential(OrderedDict({
            'downconv': nn.MaxPool2d(2, 2),
            'conv1'   : conv2dbn(32, 64, 3, 1, need_bias, act_fun),
            'conv2'   : conv2dbn(64, 64, 3, 1, need_bias, act_fun),
            'att3'    : attention(64, kind=att, reduce_ratio=reduce_ratio, kernel_size=7)
        }))  # B X 64 X H/4 X W/4
        self.downblock4 = nn.Sequential(OrderedDict({
            'downconv': nn.MaxPool2d(2, 2),
            'conv1'   : conv2dbn(64, 128, 3, 1, need_bias, act_fun),
            'conv2'   : conv2dbn(128, 128, 3, 1, need_bias, act_fun),
            'att4'    : attention(128, kind=att, reduce_ratio=reduce_ratio, kernel_size=7)
        }))  # B X 128 X H/8 X W/8
        self.bottleneck = nn.Sequential(OrderedDict({
            'downconv': nn.MaxPool2d(2, 2),
            'conv1'   : conv2dbn(128, 256, 3, 1, need_bias, act_fun),
            'conv2'   : conv2dbn(256, 256, 3, 1, need_bias, act_fun),
            'upconv'  : nn.Upsample(scale_factor=2, mode='bilinear')
        }))  # B X 128 X H/8 X W/8
        
        self.upblock4 = nn.Sequential(OrderedDict({
            'conv1' : conv2dbn(256 + 128, 128, 3, 1, need_bias, act_fun),
            'conv2' : conv2dbn(128, 128, 3, 1, need_bias, act_fun),
            'att5'  : attention(128, kind=att, reduce_ratio=reduce_ratio, kernel_size=7),
            'upconv': nn.Upsample(scale_factor=2, mode='bilinear')
        }))
        
        self.upblock3 = nn.Sequential(OrderedDict({
            'conv1' : conv2dbn(128 + 64, 64, 3, 1, need_bias, act_fun),
            'conv2' : conv2dbn(64, 64, 3, 1, need_bias, act_fun),
            'att6'  : attention(64, kind=att, reduce_ratio=reduce_ratio, kernel_size=7),
            'upconv': nn.Upsample(scale_factor=2, mode='bilinear')
        }))
        
        self.upblock2 = nn.Sequential(OrderedDict({
            'conv1' : conv2dbn(64 + 32, 32, 3, 1, need_bias, act_fun),
            'conv2' : conv2dbn(32, 32, 3, 1, need_bias, act_fun),
            'att7'  : attention(32, kind=att, reduce_ratio=reduce_ratio, kernel_size=7),
            'upconv': nn.Upsample(scale_factor=2, mode='bilinear'),
        }))
        
        self.upblock1 = nn.Sequential(OrderedDict({
            'conv1': conv2dbn(32 + 16, 16, 3, 1, need_bias, act_fun),
            'conv2': conv2dbn(16, 16, 3, 1, need_bias, act_fun),
            'att8' : attention(16, kind=att, reduce_ratio=reduce_ratio, kernel_size=7)
        }))
        
        self.outblock = nn.Conv2d(16, fout, 3, 1, 1)
    
    def forward(self, x):
        down1 = self.downblock1(x)
        down2 = self.downblock2(down1)
        down3 = self.downblock3(down2)
        down4 = self.downblock4(down3)
        up4 = self.bottleneck(down4)
        up3 = self.upblock4(torch.cat((down4, up4), dim=1))
        up2 = self.upblock3(torch.cat((down3, up3), dim=1))
        up1 = self.upblock2(torch.cat((down2, up2), dim=1))
        out = self.outblock(self.upblock1(torch.cat((down1, up1), dim=1)))
        
        return out


class AttMulResUnet2D(nn.Module):
    """
        The attention multi-resolution network
    """
    
    def __init__(self, num_input_channels=1,
                 num_output_channels=3,
                 num_channels_down=[16, 32, 64, 128, 256],
                 alpha=1.67,
                 last_act_fun=None,
                 need_bias=True,
                 upsample_mode='nearest',
                 act_fun='LeakyReLU',
                 dropout=0.):

        super(AttMulResUnet2D, self).__init__()
        n_scales = len(num_channels_down)
        
        if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
            upsample_mode = [upsample_mode] * n_scales
        
        input_depths = [num_input_channels]
        
        for i in range(n_scales):
            mrb = Block2d(num_channels_down[i], input_depths[-1],
                          alpha=alpha, act_fun=act_fun, bias=need_bias, drop=dropout)
            input_depths.append(mrb.out_dim)
            setattr(self, 'down_mb%d' % (i + 1), mrb)
        
        for i in range(1, n_scales):
            setattr(self, 'down%d' % i, nn.Sequential(*[
                conv(input_depths[i], input_depths[i], 3, stride=2, bias=need_bias),
                nn.BatchNorm2d(input_depths[i]),
                get_activation(act_fun),
                nn.Dropout2d(dropout),
            ]))
            mrb = Block2d(num_channels_down[-(i + 1)], input_depths[-i] + input_depths[-(i + 1)],
                          alpha=alpha, act_fun=act_fun, bias=need_bias, drop=dropout)
            setattr(self, 'up_mb%d' % i, mrb)
            setattr(self, 'att%d' % i,
                    GridAttentionBlock(input_depths[-i],
                                       input_depths[-(i + 1)], num_channels_down[-i]))
            setattr(self, 'up%d' % i, nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        if isinstance(last_act_fun, str) and last_act_fun.lower() == 'none':
            last_act_fun = None
        if last_act_fun is not None:
            self.outconv = nn.Sequential(*[
                conv(input_depths[1], num_output_channels, 1, 1, bias=need_bias),
                get_activation(last_act_fun)])
        else:
            self.outconv = conv(input_depths[1], num_output_channels, 1, 1, bias=need_bias)
    
    def forward(self, inp):
        x1 = self.down_mb1(inp)
        x2 = self.down_mb2(self.down1(x1))
        x3 = self.down_mb3(self.down2(x2))
        x4 = self.down_mb4(self.down3(x3))
        x5 = self.down_mb5(self.down4(x4))
        
        x4 = self.up_mb1(concat([self.att1(x5, x4), self.up1(x5)]))
        x3 = self.up_mb2(concat([self.att2(x4, x3), self.up2(x4)]))
        x2 = self.up_mb3(concat([self.att3(x3, x2), self.up3(x3)]))
        x1 = self.up_mb4(concat([self.att4(x2, x1), self.up4(x2)]))
        
        return self.outconv(x1)


__all__ = [
    "AttMulResUnet2D",
]
