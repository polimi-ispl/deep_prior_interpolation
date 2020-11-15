import torch
import torch.nn as nn
from .base import Symmetry


class Partial3DConvLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, bn=True, bias=False, sample="none-3", activation="relu"):
        super(Partial3DConvLayer, self).__init__()
        self.bn = bn
        
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
        
        if bn:
            # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
            # Applying BatchNorm2d layer after Conv will remove the channel mean
            self.batch_normalization = nn.BatchNorm3d(out_channels)
        
        if activation == "relu":
            # Used between all encoding layers
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            # Used between all decoding layers (Leaky RELU with alpha = 0.2)
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
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
            output = self.batch_normalization(output)
        
        if hasattr(self, 'activation'):
            output = self.activation(output)
        
        return output, new_mask


class PartialConvEncoderBlock(nn.Module):
    
    def __init__(self, input_channel, out_channels):
        super(PartialConvEncoderBlock, self).__init__()
        self.partialconv = Partial3DConvLayer(input_channel, out_channels, bn=True, activation='leaky_relu')
        self.down = nn.Conv3d(out_channels, out_channels, 3, 2, 1, bias=False)
    
    def forward(self, x, mask):
        x, mask = self.partialconv(x, mask)
        return self.down(x), self.down(mask)


class PartialConv3DUNet(nn.Module):
    
    # 256 x 256 image input, 256 = 2^8
    def __init__(self, input_channel, out_channels):
        super(PartialConv3DUNet, self).__init__()
        self.layers = 5
        
        # ======================= ENCODING LAYERS =======================
        # input_cxHxW --> 48xH/2xW/2
        self.encoder_1 = PartialConvEncoderBlock(input_channel, 48)
        
        # 48xH/2xW/2 --> 48xH/4xW/4
        self.encoder_2 = PartialConvEncoderBlock(48, 48)
        
        # 48xH/4xW/4 --> 48xH/8xW/8
        self.encoder_3 = PartialConvEncoderBlock(48, 48)
        
        # 48xH/8xW/8 --> 48xH/16xW/16
        self.encoder_4 = PartialConvEncoderBlock(48, 48)
        
        # 48xH/16xW/16 --> 48xH/32xW/32
        self.encoder_5 = PartialConvEncoderBlock(48, 48)
        
        # 48xH/32xW/32 --> 48xH/16xW/16
        self.decoder_5 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # ======================= DECODING LAYERS =======================
        
        # UP(48xH/16xW/16) + 48xH/16xW/16(enc_4 output) = 96xH/16xW/16 --> 96xH/8xW/8
        self.decoder_4 = nn.Sequential(
            *[nn.Conv3d(96, 96, 3, 1, 1, bias=False),
              nn.Conv3d(96, 96, 3, 1, 1, bias=False),
              nn.Upsample(scale_factor=2, mode='nearest')])
        
        # UP(96xH/8xW/8) + 48xH/8xW/8(enc_3 output) = 144xH/8xW/8 --> 96xH/4xW/4
        self.decoder_3 = nn.Sequential(
            *[nn.Conv3d(144, 96, 3, 1, 1, bias=False),
              nn.Conv3d(96, 96, 3, 1, 1, bias=False),
              nn.Upsample(scale_factor=2, mode='nearest')])
        
        # UP(96xH/4xW/4) + 48xH/4xW/4(enc_2 output) = 144xH/4xW/4 --> 96xH/2xW/2
        self.decoder_2 = nn.Sequential(
            *[nn.Conv3d(144, 96, 3, 1, 1, bias=False),
              nn.Conv3d(96, 96, 3, 1, 1, bias=False),
              nn.Upsample(scale_factor=2, mode='nearest')])
        
        # UP(96xH/2xW/2) + 48xH/2xW/2(enc_1 output) = 144xH/2xW/2 --> 96xHxW
        self.decoder_1 = nn.Sequential(
            *[nn.Conv3d(144, 96, 3, 1, 1, bias=False),
              nn.Conv3d(96, 96, 3, 1, 1, bias=False),
              nn.Upsample(scale_factor=2, mode='nearest')])
        
        # UP(96xHxW) + input_cxH/2xW/2(input) = 96 + CxHxW --> 64xHxW --> 32xHxW --> CxHxW
        self.outconv = nn.Sequential(
            *[nn.Conv3d(96 + input_channel, 96, 3, 1, 1, bias=False),
              nn.Conv3d(96, 64, 3, 1, 1, bias=False),
              nn.Conv3d(64, 32, 3, 1, 1, bias=False),
              nn.Conv3d(32, out_channels, 3, 1, 1, bias=False)])
        
        self.symmetry = Symmetry()
    
    def forward(self, input_x, mask):
        down1, mask1 = self.encoder_1(input_x, mask)
        down2, mask2 = self.encoder_2(down1, mask1)
        down3, mask3 = self.encoder_3(down2, mask2)
        down4, mask4 = self.encoder_4(down3, mask3)
        down5, _ = self.encoder_5(down4, mask4)
        
        up4 = self.decoder_5(down5)
        up3 = self.decoder_4(torch.cat([down4, up4], dim=1))
        up2 = self.decoder_3(torch.cat([down3, up3], dim=1))
        up1 = self.decoder_2(torch.cat([down2, up2], dim=1))
        up0 = self.decoder_1(torch.cat([down1, up1], dim=1))
        
        out = self.outconv(torch.cat([input_x, up0], dim=1))
        
        return self.symmetry(out)


if __name__ == '__main__':
    dtype = torch.cuda.FloatTensor
    size = (1, 64, 128, 128, 128)
    inp = torch.ones(size).type(dtype)
    input_mask = torch.ones(size).type(dtype)
    input_mask[:, :, 100:, :, :][:, :, :, 100:, :] = 0
    
    # block = PartialConvEncoderBlock(3, 48)
    # b1, m1 = block(inp, input_mask)
    # print(b1.shape)
    # print(m1.shape)
    
    conv = PartialConv3DUNet(64, 1).type(dtype)
    # l1 = nn.L1Loss()
    # inp.requires_grad = True
    
    output = conv(inp, input_mask)
    print(output.shape)
# loss = l1(output, torch.randn(1, 3, 256, 256))
# loss.backward()

# assert (torch.sum(inp.grad != inp.grad).item() == 0)
# assert (torch.sum(torch.isnan(conv.decoder_1.input_conv.weight.grad)).item() == 0)
# assert (torch.sum(torch.isnan(conv.decoder_1.input_conv.bias.grad)).item() == 0)


__all__ = [
    "Partial3DConvLayer",
    "PartialConvEncoderBlock",
    "PartialConv3DUNet",
]
