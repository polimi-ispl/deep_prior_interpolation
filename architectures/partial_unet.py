import torch
import torch.nn as nn
from .base import get_activation


class Partial2DConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, bn=True, bias=False, sample="none-3", act_fun="ReLU", drop=0.):
        super(Partial2DConv, self).__init__()
        
        if sample == "down-7":
            # Kernel Size = 7, Stride = 2, Padding = 3
            self.input_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=bias)
            self.mask_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=False)
        
        elif sample == "down-5":
            self.input_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=bias)
            self.mask_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=False)
        
        elif sample == "down-3":
            self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=bias)
            self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)
        
        else:
            self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
            self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        
        nn.init.constant_(self.mask_conv.weight, 1.0)
        
        # "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
        # negative slope of leaky_relu set to 0, same as relu
        # "fan_in" preserved variance from forward pass
        nn.init.kaiming_normal_(self.input_conv.weight, a=0, mode="fan_in")
        
        for param in self.mask_conv.parameters():
            param.requires_grad = False
        
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = get_activation(act_fun)
        self.dr = nn.Dropout2d(drop)
    
    def forward(self, input_x, mask):
        # output = W^T dot (X .* M) + b
        output = self.input_conv(input_x * mask)
        
        # requires_grad = False
        with torch.no_grad():
            # mask = (1 dot M) + 0 = M
            output_mask = self.mask_conv(mask)
        
        if self.input_conv.bias is not None:
            # spreads existing bias values out along 2nd dimension (channels) and then expands to output size
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)
        
        # mask_sum is the sum of the binary mask at every partial convolution location
        mask_is_zero = (output_mask == 0)
        # temporarily sets zero values to one to ease output calculation
        mask_sum = output_mask.masked_fill_(mask_is_zero, 1.0)
        
        # output at each location as follows:
        # output = (W^T dot (X .* M) + b - b) / M_sum + b ; if M_sum > 0
        # output = 0 ; if M_sum == 0
        output = (output - output_bias) / mask_sum + output_bias
        output = output.masked_fill_(mask_is_zero, 0.0)
        
        # mask is updated at each location
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(mask_is_zero, 0.0)
        
        if self.bn:
            output = self.bn(output)
        
        if hasattr(self, 'act'):
            output = self.act(output)
        
        output = self.dr(output)
        
        return output, new_mask


class Partial3DConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, bn=True, bias=False, sample="none-3", act_fun="ReLU", drop=0.):
        super(Partial3DConv, self).__init__()
        
        if sample == "down-7":
            # Kernel Size = 7, Stride = 2, Padding = 3
            self.input_conv = nn.Conv3d(in_channels, out_channels, 7, 2, 3, bias=bias)
            self.mask_conv = nn.Conv3d(in_channels, out_channels, 7, 2, 3, bias=False)
        
        elif sample == "down-5":
            self.input_conv = nn.Conv3d(in_channels, out_channels, 5, 2, 2, bias=bias)
            self.mask_conv = nn.Conv3d(in_channels, out_channels, 5, 2, 2, bias=False)
        
        elif sample == "down-3":
            self.input_conv = nn.Conv3d(in_channels, out_channels, 3, 2, 1, bias=bias)
            self.mask_conv = nn.Conv3d(in_channels, out_channels, 3, 2, 1, bias=False)
        
        else:
            self.input_conv = nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=bias)
            self.mask_conv = nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False)
        
        nn.init.constant_(self.mask_conv.weight, 1.0)
        
        # "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
        # negative slope of leaky_relu set to 0, same as relu
        # "fan_in" preserved variance from forward pass
        nn.init.kaiming_normal_(self.input_conv.weight, a=0, mode="fan_in")
        
        for param in self.mask_conv.parameters():
            param.requires_grad = False
        
        self.bn = nn.BatchNorm3d(out_channels) if bn else None
        self.act = get_activation(act_fun)
        self.dr = nn.Dropout2d(drop)
    
    def forward(self, input_x, mask):
        # output = W^T dot (X .* M) + b
        output = self.input_conv(input_x * mask)
        
        # requires_grad = False
        with torch.no_grad():
            # mask = (1 dot M) + 0 = M
            output_mask = self.mask_conv(mask)
        
        if self.input_conv.bias is not None:
            # spreads existing bias values out along 2nd dimension (channels) and then expands to output size
            output_bias = self.input_conv.bias.view(1, -1, 1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)
        
        # mask_sum is the sum of the binary mask at every partial convolution location
        mask_is_zero = (output_mask == 0)
        # temporarily sets zero values to one to ease output calculation
        mask_sum = output_mask.masked_fill_(mask_is_zero, 1.0)
        
        # output at each location as follows:
        # output = (W^T dot (X .* M) + b - b) / M_sum + b ; if M_sum > 0
        # output = 0 ; if M_sum == 0
        output = (output - output_bias) / mask_sum + output_bias
        output = output.masked_fill_(mask_is_zero, 0.0)
        
        # mask is updated at each location
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(mask_is_zero, 0.0)
        
        if self.bn:
            output = self.bn(output)
        
        if hasattr(self, 'act'):
            output = self.act(output)
        
        output = self.dr(output)
        
        return output, new_mask


class Partial2DBlock(nn.Module):
    
    def __init__(self, input_channel, out_channels, bn, act_fun, bias, drop):
        super(Partial2DBlock, self).__init__()
        self.partialconv = Partial2DConv(input_channel, out_channels, bn=bn, act_fun=act_fun, drop=drop)
        self.down = nn.Conv2d(out_channels, out_channels, 3, 2, 1, bias=bias)
        self.dr = nn.Dropout2d(drop)
    
    def forward(self, x, mask):
        x, mask = self.partialconv(x, mask)
        x, mask = self.down(x), self.down(mask)
        x, mask = self.dr(x), self.dr(mask)
        return x, mask


class Partial3DBlock(nn.Module):
    
    def __init__(self, input_channel, out_channels, bn, act_fun, bias, drop):
        super(Partial3DBlock, self).__init__()
        self.partialconv = Partial3DConv(input_channel, out_channels, bn=bn, act_fun=act_fun, bias=bias, drop=drop)
        self.down = nn.Conv3d(out_channels, out_channels, 3, 2, 1, bias=bias)
        self.dr = nn.Dropout3d(drop)
    
    def forward(self, x, mask):
        x, mask = self.partialconv(x, mask)
        x, mask = self.down(x), self.down(mask)
        x, mask = self.dr(x), self.dr(mask)
        return x, mask


class PartialUNet(nn.Module):
    
    def __init__(self, num_input_channels=1,
                 num_output_channels=1,
                 use_bn=True,
                 need_bias=True,
                 act_fun='LeakyReLU',
                 dropout=0.):
        super(PartialUNet, self).__init__()
        self.layers = 5
        
        self.enc1 = Partial2DBlock(num_input_channels, 48, use_bn, act_fun, need_bias, dropout)
        self.enc2 = Partial2DBlock(48, 48, use_bn, act_fun, need_bias, dropout)
        self.enc3 = Partial2DBlock(48, 48, use_bn, act_fun, need_bias, dropout)
        self.enc4 = Partial2DBlock(48, 48, use_bn, act_fun, need_bias, dropout)
        self.enc5 = Partial2DBlock(48, 48, use_bn, act_fun, need_bias, dropout)
        
        self.dec5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec4 = nn.Sequential(*[nn.Conv2d(96, 96, 3, 1, 1, bias=False),
                                    nn.Conv2d(96, 96, 3, 1, 1, bias=False),
                                    nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.Dropout2d(dropout)])
        self.dec3 = nn.Sequential(*[nn.Conv2d(144, 96, 3, 1, 1, bias=False),
                                    nn.Conv2d(96, 96, 3, 1, 1, bias=False),
                                    nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.Dropout2d(dropout)])
        self.dec2 = nn.Sequential(*[nn.Conv2d(144, 96, 3, 1, 1, bias=False),
                                    nn.Conv2d(96, 96, 3, 1, 1, bias=False),
                                    nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.Dropout2d(dropout)])
        self.dec1 = nn.Sequential(*[nn.Conv2d(144, 96, 3, 1, 1, bias=False),
                                    nn.Conv2d(96, 96, 3, 1, 1, bias=False),
                                    nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.Dropout2d(dropout)])
        
        self.last_kernel = nn.Sequential(*[nn.Conv2d(96 + num_input_channels, 96, 3, 1, 1, bias=False),
                                           nn.Conv2d(96, 64, 3, 1, 1, bias=False),
                                           nn.Conv2d(64, 32, 3, 1, 1, bias=False),
                                           nn.Conv2d(32, num_output_channels, 3, 1, 1, bias=False)])
    
    def forward(self, input_x, mask):
        down1, mask1 = self.enc1(input_x, mask)
        down2, mask2 = self.enc2(down1, mask1)
        down3, mask3 = self.enc3(down2, mask2)
        down4, mask4 = self.enc4(down3, mask3)
        down5, _     = self.enc5(down4, mask4)
        
        up4 = self.dec5(down5)
        up3 = self.dec4(torch.cat([down4, up4], dim=1))
        up2 = self.dec3(torch.cat([down3, up3], dim=1))
        up1 = self.dec2(torch.cat([down2, up2], dim=1))
        up0 = self.dec1(torch.cat([down1, up1], dim=1))
        
        out = self.last_kernel(torch.cat([input_x, up0], dim=1))
        
        return out


class PartialUNet3D(nn.Module):
    
    def __init__(self, num_input_channels=1,
                 num_output_channels=1,
                 use_bn=True,
                 need_bias=True,
                 act_fun='LeakyReLU',
                 dropout=0.):
        super(PartialUNet3D, self).__init__()
        self.layers = 5
        
        self.enc1 = Partial3DBlock(num_input_channels, 48, use_bn, act_fun, need_bias, dropout)
        self.enc2 = Partial3DBlock(48, 48, use_bn, act_fun, need_bias, dropout)
        self.enc3 = Partial3DBlock(48, 48, use_bn, act_fun, need_bias, dropout)
        self.enc4 = Partial3DBlock(48, 48, use_bn, act_fun, need_bias, dropout)
        self.enc5 = Partial3DBlock(48, 48, use_bn, act_fun, need_bias, dropout)
        
        self.dec5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec4 = nn.Sequential(*[nn.Conv3d(96, 96, 3, 1, 1, bias=False),
                                    nn.Conv3d(96, 96, 3, 1, 1, bias=False),
                                    nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.Dropout3d(dropout)])
        self.dec3 = nn.Sequential(*[nn.Conv3d(144, 96, 3, 1, 1, bias=False),
                                    nn.Conv3d(96, 96, 3, 1, 1, bias=False),
                                    nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.Dropout3d(dropout)])
        self.dec2 = nn.Sequential(*[nn.Conv3d(144, 96, 3, 1, 1, bias=False),
                                    nn.Conv3d(96, 96, 3, 1, 1, bias=False),
                                    nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.Dropout3d(dropout)])
        self.dec1 = nn.Sequential(*[nn.Conv3d(144, 96, 3, 1, 1, bias=False),
                                    nn.Conv3d(96, 96, 3, 1, 1, bias=False),
                                    nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.Dropout3d(dropout)])
        
        self.last_kernel = nn.Sequential(*[nn.Conv3d(96 + num_input_channels, 96, 3, 1, 1, bias=False),
                                           nn.Conv3d(96, 64, 3, 1, 1, bias=False),
                                           nn.Conv3d(64, 32, 3, 1, 1, bias=False),
                                           nn.Conv3d(32, num_output_channels, 3, 1, 1, bias=False)])
    
    def forward(self, input_x, mask):
        down1, mask1 = self.enc1(input_x, mask)
        down2, mask2 = self.enc2(down1, mask1)
        down3, mask3 = self.enc3(down2, mask2)
        down4, mask4 = self.enc4(down3, mask3)
        down5, _     = self.enc5(down4, mask4)
        
        up4 = self.dec5(down5)
        up3 = self.dec4(torch.cat([down4, up4], dim=1))
        up2 = self.dec3(torch.cat([down3, up3], dim=1))
        up1 = self.dec2(torch.cat([down2, up2], dim=1))
        up0 = self.dec1(torch.cat([down1, up1], dim=1))
        
        out = self.last_kernel(torch.cat([input_x, up0], dim=1))
        
        return out


__all__ = [
    "PartialUNet",
    "PartialUNet3D",
]
