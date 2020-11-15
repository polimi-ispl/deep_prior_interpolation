import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim
        
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
    
    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))
        
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
        
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(
                    inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])
        
        return torch.cat(inputs_, dim=self.dim)
    
    def __len__(self):
        return len(self._modules)


def concat(inputs):
    inputs_shapes2 = [x.shape[2] for x in inputs]
    inputs_shapes3 = [x.shape[3] for x in inputs]
    
    if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
            np.array(inputs_shapes3) == min(inputs_shapes3)):
        inputs_ = inputs
    else:
        target_shape2 = min(inputs_shapes2)
        target_shape3 = min(inputs_shapes3)
        
        inputs_ = []
        for inp in inputs:
            diff2 = (inp.size(2) - target_shape2) // 2
            diff3 = (inp.size(3) - target_shape3) // 2
            inputs_.append(
                inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])
    
    return torch.cat(inputs_, dim=1)


def act(act_fun='LeakyReLU'):
    """
    Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        elif act_fun == 'ReLU':
            return nn.ReLU()
        elif act_fun == 'Tanh':
            return nn.Tanh()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True):
    """
        The convolutional filter with same zero pad.
    """
    to_pad = int((kernel_size - 1) / 2)
    
    convolver = nn.Conv2d(in_f, out_f, kernel_size,
                          stride, padding=to_pad, bias=bias)
    
    layers = filter(lambda x: x is not None, [convolver])
    return nn.Sequential(*layers)


def conv2dbn(in_f, out_f, kernel_size, stride=1, bias=True, act_fun='LeakyReLU'):
    block = conv(in_f, out_f, kernel_size, stride=stride, bias=bias)
    block.add(bn(out_f))
    block.add(act(act_fun))
    return block


class MultiResBlock(nn.Module):
    def __init__(self, U, f_in, alpha=1.67, act_fun='LeakyReLU', bias=True):
        super(MultiResBlock, self).__init__()
        W = alpha * U
        self.out_dim = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
        self.shortcut = conv2dbn(f_in, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1,
                                 bias=bias, act_fun=act_fun)
        self.conv3x3 = conv2dbn(f_in, int(W * 0.167), 3, 1, bias=bias,
                                act_fun=act_fun)
        self.conv5x5 = conv2dbn(int(W * 0.167), int(W * 0.333), 3, 1, bias=bias,
                                act_fun=act_fun)
        self.conv7x7 = conv2dbn(int(W * 0.333), int(W * 0.5), 3, 1, bias=bias,
                                act_fun=act_fun)
        self.bn1 = bn(self.out_dim)
        self.bn2 = bn(self.out_dim)
        self.accfun = act(act_fun)
    
    def forward(self, input):
        out1 = self.conv3x3(input)
        out2 = self.conv5x5(out1)
        out3 = self.conv7x7(out2)
        out = self.bn1(torch.cat([out1, out2, out3], axis=1))
        out = torch.add(self.shortcut(input), out)
        out = self.bn2(self.accfun(out))
        return out


class PathRes(nn.Module):
    def __init__(self, f_in, f_out, length, act_fun='LeakyReLU', bias=True):
        super(PathRes, self).__init__()
        self.network = []
        self.network.append(conv2dbn(f_in, f_out, 3, 1, bias=bias, act_fun=act_fun))
        self.network.append(conv2dbn(f_in, f_out, 1, 1, bias=bias, act_fun=act_fun))
        self.network.append(bn(f_out))
        for i in range(length - 1):
            self.network.append(conv2dbn(f_out, f_out, 3, 1, bias=bias, act_fun=act_fun))
            self.network.append(conv2dbn(f_out, f_out, 1, 1, bias=bias, act_fun=act_fun))
            self.network.append(bn(f_out))
        self.accfun = act(act_fun)
        self.length = length
        self.network = nn.Sequential(*self.network)
    
    def forward(self, input):
        out = self.network[2](self.accfun(torch.add(self.network[0](input),
                                                    self.network[1](input))))
        for i in range(1, self.length):
            out = self.network[i * 3 + 2](self.accfun(torch.add(self.network[i * 3](out),
                                                                self.network[i * 3 + 1](out))))
        
        return out


class GridAttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(GridAttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            conv(F_g, F_int, 1, 1),
            bn(F_int)
        )
        
        self.W_x = nn.Sequential(
            conv(F_l, F_int, 3, 2),
            bn(F_int)
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


def MulResUnet(
        num_input_channels=1, num_output_channels=1,
        num_channels_down=[16, 32, 64, 128, 256], num_channels_up=[16, 32, 64, 128, 256],
        num_channels_skip=[16, 32, 64, 128], alpha=1.67,
        need_sigmoid=True, need_bias=True,
        upsample_mode='nearest', act_fun='LeakyReLU'):
    """ The 2D multi-resolution Unet

    Arguments:
        num_input_channels (int) -- The channels of the input data.
        num_output_channels (int) -- The channels of the output data.
        num_channels_down (list) -- The channels of differnt layer in the encoder of networks.
        num_channels_up (list) -- The channels of differnt layer in the decoder of networks.
        num_channels_skip (list) -- The channels of path residual block corresponding to different layer.
        alpha (float) -- the value multiplying to the number of filters.
        need_sigmoid (Bool) -- if add the sigmoid layer in the last of decoder.
        need_bias (Bool) -- If add the bias in every convolutional filters.
        upsample_mode (str) -- The type of upsampling in the decoder, including 'bilinear' and 'nearest'.
        act_fun (str) -- The activate function, including LeakyReLU, ReLU, Tanh, ELU.
    """
    assert len(num_channels_down) == len(
        num_channels_up) == (len(num_channels_skip) + 1)
    
    n_scales = len(num_channels_down)
    
    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    
    last_scale = n_scales - 1
    
    model = nn.Sequential()
    model_tmp = model
    multires = MultiResBlock(num_channels_down[0], num_input_channels,
                             alpha=alpha, act_fun=act_fun, bias=need_bias)
    
    model_tmp.add(multires)
    input_depth = multires.out_dim
    
    for i in range(1, len(num_channels_down)):
        
        deeper = nn.Sequential()
        skip = nn.Sequential()
        # multi-res Block in the encoders
        multires = MultiResBlock(num_channels_down[i], input_depth,
                                 alpha=alpha, act_fun=act_fun, bias=need_bias)
        # stride downsampling.
        deeper.add(conv(input_depth, input_depth, 3, stride=2, bias=need_bias))
        deeper.add(bn(input_depth))
        deeper.add(act(act_fun))
        deeper.add(multires)
        
        if num_channels_skip[i - 1] != 0:
            # add the path residual block, note that the number of filters is set to 1.
            skip.add(PathRes(input_depth, num_channels_skip[i - 1], 1, act_fun=act_fun, bias=need_bias))
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        deeper_main = nn.Sequential()
        
        if i != len(num_channels_down) - 1:
            # not the deepest
            deeper.add(deeper_main)
        # add upsampling to the decoder
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        # add multi-res block to the decoder
        model_tmp.add(MultiResBlock(num_channels_up[i - 1], multires.out_dim + num_channels_skip[i - 1],
                                    alpha=alpha, act_fun=act_fun, bias=need_bias))
        
        input_depth = multires.out_dim
        model_tmp = deeper_main
    W = num_channels_up[0] * alpha
    last_kernal = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
    # add the convolutional filter for output.
    model.add(
        conv(last_kernal, num_output_channels, 1, bias=need_bias))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    return model


class AttMulResUnet2D(nn.Module):
    """
        The attention multi-resolution network
    """
    
    def __init__(self, num_input_channels=1, num_output_channels=3,
                 num_channels_down=[16, 32, 64, 128, 256],
                 alpha=1.67, need_sigmoid=False, need_bias=True,
                 upsample_mode='nearest', act_fun='LeakyReLU') -> None:
        """
            The 3D multi-resolution Unet
        Arguments:
            num_input_channels (int) -- The channels of the input data.
            num_output_channels (int) -- The channels of the output data.
            num_channels_down (list) -- The channels of differnt layer in the encoder of networks.
            num_channels_up (list) -- The channels of differnt layer in the decoder of networks.
            num_channels_skip (list) -- The channels of path residual block corresponding to different layer.
            alpha (float) -- the value multiplying to the number of filters.
            need_sigmoid (Bool) -- if add the sigmoid layer in the last of decoder.
            need_bias (Bool) -- If add the bias in every convolutional filters.
            upsample_mode (str) -- The type of upsampling in the decoder, including 'bilinear' and 'nearest'.
            act_fun (str) -- The activate function, including LeakyReLU, ReLU, Tanh, ELU.
        """
        super(AttMulResUnet2D, self).__init__()
        n_scales = len(num_channels_down)
        
        if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
            upsample_mode = [upsample_mode] * n_scales
        
        input_depths = [num_input_channels]
        
        for i in range(n_scales):
            mrb = MultiResBlock(num_channels_down[i], input_depths[-1])
            input_depths.append(mrb.out_dim)
            setattr(self, 'down_mb%d' % (i + 1), mrb)
        
        for i in range(1, n_scales):
            setattr(self, 'down%d' % i, nn.Sequential(*[
                conv(input_depths[i], input_depths[i], 3, stride=2, bias=need_bias),
                bn(input_depths[i]),
                act(act_fun)
            ]))
            mrb = MultiResBlock(num_channels_down[-(i + 1)], input_depths[-i] + input_depths[-(i + 1)])
            setattr(self, 'up_mb%d' % i, mrb)
            setattr(self, 'att%d' % i,
                    GridAttentionBlock(input_depths[-i],
                                       input_depths[-(i + 1)], num_channels_down[-i]))
            setattr(self, 'up%d' % i, nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        if need_sigmoid:
            self.outconv = nn.Sequential(*[
                conv(input_depths[1], num_output_channels, 1, 1, bias=need_bias),
                nn.Sigmoid()])
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


class ChannelGata(nn.Module):
    """
        The channel block. process the feature extracted by encoder and decoder.
        (Convolutional block attention module)
        referenc (squeeze-and-excitation network)
    """
    
    def __init__(self, f_x, reduction_ratio=4):
        super(ChannelGata, self).__init__()
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
        (Convolutional block attention module)
        referenc (squeeze-and-excitation network)
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
        self.ChannelGate = ChannelGata(f_x, reduction_ratio)
        self.SpatialGate = SpatialGate(f_x, kernel_size=kernel_size)
    
    def forward(self, x):
        return self.SpatialGate(self.ChannelGate(x))


class Identidy(nn.Module):
    def __init__(self):
        super(Identidy, self).__init__()
    
    def forward(self, x):
        return x


def attention(f_x, kind='unet', reduce_ratio=8, kernel_size=7):
    if kind == 'cbam':
        return CBAM(f_x, reduction_ratio=reduce_ratio, kernel_size=kernel_size)
    else:
        return Identidy()


class AttentionUnet(nn.Module):
    def __init__(self, fin=3, fout=1, act_fun='LeakyReLU', need_bias=True, att='cbam', reduce_ratio=4):
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


# For the 3D network
class Concat3D(nn.Module):
    def __init__(self, dim, *args):
        super(Concat3D, self).__init__()
        self.dim = dim
        
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
    
    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))
        
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
        inputs_shapes4 = [x.shape[4] for x in inputs]
        
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)) and np.all(
            np.array(inputs_shapes4) == min(inputs_shapes4)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            target_shape4 = min(inputs_shapes4)
            
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                diff4 = (inp.size(4) - target_shape4) // 2
                inputs_.append(
                    inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3, diff4:diff4 + target_shape4])
        
        return torch.cat(inputs_, dim=self.dim)
    
    def __len__(self):
        return len(self._modules)


def conv3d(in_f, out_f, kernel_size, stride=1, bias=True):
    """
        The 3D convolutional filters with kind of stride, avg pooling, max pooling.
        Note that the padding is zero padding.
    """
    to_pad = int((kernel_size - 1) / 2)
    
    convolver = nn.Conv3d(in_f, out_f, kernel_size,
                          stride, padding=to_pad, bias=bias)
    
    layers = filter(lambda x: x is not None, [convolver])
    return nn.Sequential(*layers)


def conv3dbn(in_f, out_f, kernel_size=3, stride=1, bias=True, act_fun='LeakyReLU'):
    block = []
    block.append(conv3d(in_f, out_f, kernel_size, stride=stride, bias=bias))
    block.append(nn.BatchNorm3d(out_f))
    block.append(act(act_fun))
    return nn.Sequential(*block)


class MultiRes3dBlock(nn.Module):
    def __init__(self, U, f_in, alpha=1.67, act_fun='LeakyReLU', bias=True):
        super(MultiRes3dBlock, self).__init__()
        W = alpha * U
        self.out_dim = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
        self.shortcut = conv3dbn(f_in, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1,
                                 bias=bias, act_fun=act_fun)
        self.conv3x3 = conv3dbn(f_in, int(W * 0.167), 3, 1, bias=bias,
                                act_fun=act_fun)
        self.conv5x5 = conv3dbn(int(W * 0.167), int(W * 0.333), 3, 1, bias=bias,
                                act_fun=act_fun)
        self.conv7x7 = conv3dbn(int(W * 0.333), int(W * 0.5), 3, 1, bias=bias,
                                act_fun=act_fun)
        self.bn1 = nn.BatchNorm3d(self.out_dim)
        self.bn2 = nn.BatchNorm3d(self.out_dim)
        self.accfun = act(act_fun)
    
    def forward(self, input):
        out1 = self.conv3x3(input)
        out2 = self.conv5x5(out1)
        out3 = self.conv7x7(out2)
        out = self.bn1(torch.cat([out1, out2, out3], axis=1))
        out = torch.add(self.shortcut(input), out)
        out = self.bn2(self.accfun(out))
        return out


class PathRes3d(nn.Module):
    def __init__(self, f_in, f_out, act_fun='LeakyReLU', bias=True):
        super(PathRes3d, self).__init__()
        self.conv3x3 = conv3dbn(f_in, f_out, 3, 1, bias=bias, act_fun=act_fun)
        self.conv1x1 = conv3dbn(f_in, f_out, 1, 1, bias=bias, act_fun=act_fun)
        self.bn = nn.BatchNorm3d(f_out)
        self.accfun = act(act_fun)
    
    def forward(self, input):
        out = self.bn(self.accfun(torch.add(self.conv1x1(input),
                                            self.conv3x3(input))))
        return out


class Symmetry(nn.Module):
    def __init__(self):
        super(Symmetry, self).__init__()
    
    def forward(self, x):
        return (x + x.transpose(-2, -1)) / 2


def MulResUnet3D(
        num_input_channels=1, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 256], num_channels_up=[16, 32, 64, 128, 256],
        num_channels_skip=[16, 32, 64, 128],
        alpha=1.67, need_sigmoid=False, need_bias=True,
        upsample_mode='nearest', act_fun='LeakyReLU'):
    """
        The 3D multi-resolution Unet
    Arguments:
        num_input_channels (int) -- The channels of the input data.
        num_output_channels (int) -- The channels of the output data.
        num_channels_down (list) -- The channels of differnt layer in the encoder of networks.
        num_channels_up (list) -- The channels of differnt layer in the decoder of networks.
        num_channels_skip (list) -- The channels of path residual block corresponding to different layer.
        alpha (float) -- the value multiplying to the number of filters.
        need_sigmoid (Bool) -- if add the sigmoid layer in the last of decoder.
        need_bias (Bool) -- If add the bias in every convolutional filters.
        upsample_mode (str) -- The type of upsampling in the decoder, including 'bilinear' and 'nearest'.
        act_fun (str) -- The activate function, including LeakyReLU, ReLU, Tanh, ELU.
    """
    assert len(num_channels_down) == len(
        num_channels_up) == (len(num_channels_skip) + 1)
    
    n_scales = len(num_channels_down)
    
    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    
    last_scale = n_scales - 1
    
    model = nn.Sequential()
    model_tmp = model
    multires = MultiRes3dBlock(num_channels_down[0], num_input_channels,
                               alpha=alpha, act_fun=act_fun, bias=need_bias)
    
    model_tmp.add(multires)
    input_depth = multires.out_dim
    
    for i in range(1, len(num_channels_down)):
        
        deeper = nn.Sequential()
        skip = nn.Sequential()
        # add the multi-res block for encoder
        multires = MultiRes3dBlock(num_channels_down[i], input_depth,
                                   alpha=alpha, act_fun=act_fun, bias=need_bias)
        # add the stride downsampling for encoder
        deeper.add(conv3d(input_depth, input_depth, 3, stride=2, bias=need_bias))
        deeper.add(nn.BatchNorm3d(input_depth))
        deeper.add(act(act_fun))
        deeper.add(multires)
        
        if num_channels_skip[i - 1] != 0:
            # add the Path residual block with skip-connection
            skip.add(PathRes3d(input_depth, num_channels_skip[i - 1], act_fun=act_fun, bias=need_bias))
            model_tmp.add(Concat3D(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        deeper_main = nn.Sequential()
        
        if i != len(num_channels_down) - 1:
            deeper.add(deeper_main)
        # add the upsampling to decoder
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        # add the multi-res block to decoder
        model_tmp.add(MultiRes3dBlock(num_channels_up[i - 1], multires.out_dim + num_channels_skip[i - 1],
                                      alpha=alpha, act_fun=act_fun, bias=need_bias))
        
        input_depth = multires.out_dim
        model_tmp = deeper_main
    W = num_channels_up[0] * alpha
    last_kernal = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
    # convolutional filter for output
    model.add(
        conv3d(last_kernal, num_output_channels, 3, bias=need_bias))
    if need_sigmoid:
        model.add(Symmetry())
    
    return model


__all__ = [
    "MulResUnet",
    "AttMulResUnet2D",
    "Symmetry",
    "MulResUnet3D",
]
