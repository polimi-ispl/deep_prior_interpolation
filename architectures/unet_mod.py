import torch
import torch.nn as nn
from .base import conv_mod, conv, ListModule, get_activation, conv2dbn

__all__ = [
    "Unet",
]


class unetConv(nn.Module):
    def __init__(self, in_size, out_size, bias, act_fun, drop=0.):
        super(unetConv, self).__init__()
        
        self.conv1 = conv2dbn(in_size, out_size, 3, bias=bias, act_fun=act_fun)
        self.conv2 = conv2dbn(out_size, out_size, 3, bias=bias, act_fun=act_fun)
        
        self.dr = nn.Dropout2d(drop)
    
    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.dr(out)
        out = self.conv2(out)
        out = self.dr(out)
        return out


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, bias, act_fun, drop=0.):
        super(unetDown, self).__init__()
        self.conv = unetConv(in_size, out_size, bias, act_fun)
        self.down = nn.MaxPool2d(2, 2)
        self.dr = nn.Dropout2d(drop)
    
    def forward(self, inputs):
        out = self.down(inputs)
        out = self.conv(out)
        out = self.dr(out)
        return out


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, bias, act_fun, drop=0., same_num_filt=False):
        super(unetUp, self).__init__()
        
        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
        elif upsample_mode == 'bilinear' or upsample_mode == 'nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    conv(num_filt, out_size, 3, bias=bias))
        else:
            assert False
        self.conv = unetConv(out_size * 2, out_size, bias, act_fun, drop)
        self.dr = nn.Dropout2d(drop)
    
    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)
        
        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2: diff2 + in1_up.size(2), diff3: diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2
        
        output = self.conv(torch.cat([in1_up, inputs2_], 1))
        output = self.dr(output)
        return output


class Unet(nn.Module):

    def __init__(self, num_input_channels=1, num_output_channels=1,
                 filters=[16, 32, 64, 128, 256],
                 act_fun='ReLU', upsample_mode='deconv', dropout=0.,
                 last_act_fun=None, need_bias=True):
        super(Unet, self).__init__()
        
        self.num_layers = len(filters)
        
        self.start = unetConv(num_input_channels, filters[0], need_bias, act_fun, dropout)
        
        self.down_blocks = []  # ordered as they are applied
        self.up_blocks = []  # ordered as the are applied
        for i in range(self.num_layers-1):
            self.down_blocks.append(unetDown(filters[i], filters[i+1], need_bias, act_fun, dropout))
            self.up_blocks.append(unetUp(filters[self.num_layers - 1 - i], upsample_mode, need_bias, act_fun, dropout))

        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias)
        
        if isinstance(last_act_fun, str) and last_act_fun.lower() == 'none':
            last_act_fun = None
        if last_act_fun is not None:
            self.final = nn.Sequential(self.final, get_activation(last_act_fun))
    
    def forward(self, inputs):
        
        out = [inputs]
        for i in range(self.num_layers-1):
            print(i)
            if i == 0:
                out += self.start(out[-1])
            else:
                out += self.down_blocks[i](out[-1])
        
        up = [out[-1]]
        for i in range(self.num_layers-1):
            up += self.up_blocks[i](up[self.num_layers-i], out[self.num_layers-1-i])
        
        return self.final(up[-1])
