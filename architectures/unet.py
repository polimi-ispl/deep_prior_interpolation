import torch
import torch.nn as nn
from .base import conv_mod, ListModule, act


class unetConv(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad, act_fun, drop=0.):
        super(unetConv, self).__init__()

        # print(pad)
        if norm_layer is not None:
            self.conv1 = nn.Sequential(conv_mod(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       act_fun,)
            self.conv2 = nn.Sequential(conv_mod(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       act_fun,)
        else:
            self.conv1 = nn.Sequential(conv_mod(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       act_fun,)
            self.conv2 = nn.Sequential(conv_mod(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       act_fun,)
        
        self.dr = nn.Dropout2d(drop)
        
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.dr(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dr(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad, act_fun, drop=0.):
        super(unetDown, self).__init__()
        self.conv = unetConv(in_size, out_size, norm_layer, need_bias, pad, act_fun)
        self.down = nn.MaxPool2d(2, 2)
        self.dr = nn.Dropout2d(drop)

    def forward(self, inputs):
        outputs = self.down(inputs)
        outputs = self.dr(outputs)
        outputs = self.conv(outputs)
        outputs = self.dr(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, act_fun, drop=0., same_num_filt=False):
        super(unetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv = unetConv(out_size * 2, out_size, None, need_bias, pad, act_fun, drop)
        elif upsample_mode == 'bilinear' or upsample_mode == 'nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    conv_mod(num_filt, out_size, 3, bias=need_bias, pad=pad))
            self.conv = unetConv(out_size * 2, out_size, None, need_bias, pad, act_fun, drop)
        else:
            assert False
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
    
    
class UNet(nn.Module):
    """
        upsample_mode in ['deconv', 'nearest', 'linear']
        pad in ['zero', 'replication', 'none']
    """

    def __init__(self, num_input_channels=1, num_output_channels=1,
                 filters=[16, 32, 64, 128, 256], more_layers=0, concat_x=False,
                 act_fun='ReLU', upsample_mode='deconv', pad='zero', dropout=0.,
                 norm_layer=nn.InstanceNorm2d, last_act_fun=None, need_bias=True):
        super(UNet, self).__init__()

        self.more_layers = more_layers
        self.concat_x = concat_x

        act_fun = act(act_fun)
        
        self.start = unetConv(num_input_channels, filters[0] if not concat_x else filters[0] - num_input_channels,
                              norm_layer, need_bias, pad, act_fun, dropout)

        self.down1 = unetDown(filters[0], filters[1] if not concat_x else filters[1] - num_input_channels, norm_layer,
                              need_bias, pad, act_fun, dropout)
        self.down2 = unetDown(filters[1], filters[2] if not concat_x else filters[2] - num_input_channels, norm_layer,
                              need_bias, pad, act_fun, dropout)
        self.down3 = unetDown(filters[2], filters[3] if not concat_x else filters[3] - num_input_channels, norm_layer,
                              need_bias, pad, act_fun, dropout)
        self.down4 = unetDown(filters[3], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer,
                              need_bias, pad, act_fun, dropout)
        
        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [unetDown(filters[4], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer,
                                        need_bias, pad, act_fun, dropout)
                               for _ in range(self.more_layers)]
            self.more_ups = [unetUp(filters[4], upsample_mode, need_bias, pad, act_fun, dropout, same_num_filt=True)
                             for _ in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups = ListModule(*self.more_ups)

        self.up4 = unetUp(filters[3], upsample_mode, need_bias, pad, act_fun, dropout)
        self.up3 = unetUp(filters[2], upsample_mode, need_bias, pad, act_fun, dropout)
        self.up2 = unetUp(filters[1], upsample_mode, need_bias, pad, act_fun, dropout)
        self.up1 = unetUp(filters[0], upsample_mode, need_bias, pad, act_fun, dropout)

        self.final = conv_mod(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)

        if isinstance(last_act_fun, str) and last_act_fun.lower() == 'none':
            last_act_fun = None
        if last_act_fun is not None:
            self.final = nn.Sequential(self.final, act(last_act_fun))

    def forward(self, inputs):

        # Downsample
        downs = [inputs]
        down = nn.AvgPool2d(2, 2)
        for i in range(4 + self.more_layers):
            downs.append(down(downs[-1]))

        in64 = self.start(inputs)
        if self.concat_x:
            in64 = torch.cat([in64, downs[0]], 1)

        down1 = self.down1(in64)
        if self.concat_x:
            down1 = torch.cat([down1, downs[1]], 1)

        down2 = self.down2(down1)
        if self.concat_x:
            down2 = torch.cat([down2, downs[2]], 1)

        down3 = self.down3(down2)
        if self.concat_x:
            down3 = torch.cat([down3, downs[3]], 1)

        down4 = self.down4(down3)
        if self.concat_x:
            down4 = torch.cat([down4, downs[4]], 1)

        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                # print(prevs[-1].size())
                out = d(prevs[-1])
                if self.concat_x:
                    out = torch.cat([out, downs[kk + 5]], 1)

                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more - idx - 2]
                up_ = l(up_, prevs[self.more - idx - 2])
        else:
            up_ = down4

        up4 = self.up4(up_, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)

        return self.final(up1)


__all__ = [
    "UNet",
]
