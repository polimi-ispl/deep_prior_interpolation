import numpy as np
import torch
import torch.nn as nn


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
        out = self.bn1(torch.cat([out1, out2, out3], dim=1))
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
        out = self.bn1(torch.cat([out1, out2, out3], dim=1))
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


def MulResUnet3D(
        num_input_channels=1, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 256], num_channels_up=[16, 32, 64, 128, 256],
        num_channels_skip=[16, 32, 64, 128],
        alpha=1.67, need_sigmoid=True, need_bias=True,
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
        model.add(nn.Sigmoid())
    
    return model


class UNet(nn.Module):
    """
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    """
    
    def __init__(self, num_input_channels=3, num_output_channels=3,
                 filters=[16, 32, 64, 128, 256], more_layers=0, concat_x=False,
                 activation='ReLU', upsample_mode='deconv', pad='zero',
                 norm_layer=nn.InstanceNorm2d, need_sigmoid=True, need_bias=True):
        super(UNet, self).__init__()
        
        self.more_layers = more_layers
        self.concat_x = concat_x
        
        if activation == "ReLU":
            act_fun = nn.ReLU()
        elif activation == "Tanh":
            act_fun = nn.Tanh()
        elif activation == "LeakyReLU":
            act_fun = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError("Activation has to be in [ReLU, Tanh, LeakyReLU]")
        
        self.start = unetConv2(num_input_channels, filters[0] if not concat_x else filters[0] - num_input_channels,
                               norm_layer, need_bias, pad, act_fun)
        
        self.down1 = unetDown(filters[0], filters[1] if not concat_x else filters[1] - num_input_channels, norm_layer,
                              need_bias, pad, act_fun)
        self.down2 = unetDown(filters[1], filters[2] if not concat_x else filters[2] - num_input_channels, norm_layer,
                              need_bias, pad, act_fun)
        self.down3 = unetDown(filters[2], filters[3] if not concat_x else filters[3] - num_input_channels, norm_layer,
                              need_bias, pad, act_fun)
        self.down4 = unetDown(filters[3], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer,
                              need_bias, pad, act_fun)
        
        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                unetDown(filters[4], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer,
                         need_bias, pad, act_fun) for i in range(self.more_layers)]
            self.more_ups = [unetUp(filters[4], upsample_mode, need_bias, pad, act_fun, same_num_filt=True) for i in
                             range(self.more_layers)]
            
            self.more_downs = ListModule(*self.more_downs)
            self.more_ups = ListModule(*self.more_ups)
        
        self.up4 = unetUp(filters[3], upsample_mode, need_bias, pad, act_fun)
        self.up3 = unetUp(filters[2], upsample_mode, need_bias, pad, act_fun)
        self.up2 = unetUp(filters[1], upsample_mode, need_bias, pad, act_fun)
        self.up1 = unetUp(filters[0], upsample_mode, need_bias, pad, act_fun)
        
        self.final = conv(filters[0], num_output_channels,
                          1, bias=need_bias, pad=pad)
        
        if need_sigmoid:
            self.final = nn.Sequential(self.final, nn.Sigmoid())
    
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


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad, act_fun):
        super(unetConv2, self).__init__()
        
        # print(pad)
        if norm_layer is not None:
            self.conv1 = nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size), act_fun, )
            self.conv2 = nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size), act_fun, )
        else:
            self.conv1 = nn.Sequential(
                conv(in_size, out_size, 3, bias=need_bias, pad=pad), act_fun, )
            self.conv2 = nn.Sequential(
                conv(out_size, out_size, 3, bias=need_bias, pad=pad), act_fun, )
    
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad, act_fun):
        super(unetDown, self).__init__()
        self.conv = unetConv2(
            in_size, out_size, norm_layer, need_bias, pad, act_fun)
        self.down = nn.MaxPool2d(2, 2)
    
    def forward(self, inputs):
        outputs = self.down(inputs)
        outputs = self.conv(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, act_fun, same_num_filt=False):
        super(unetUp, self).__init__()
        
        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(
                num_filt, out_size, 4, stride=2, padding=1)
            self.conv = unetConv2(out_size * 2, out_size,
                                  None, need_bias, pad, act_fun)
        elif upsample_mode == 'bilinear' or upsample_mode == 'nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    conv(num_filt, out_size, 3, bias=need_bias, pad=pad))
            self.conv = unetConv2(out_size * 2, out_size,
                                  None, need_bias, pad, act_fun)
        else:
            assert False
    
    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)
        
        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2: diff2 +
                                            in1_up.size(2), diff3: diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2
        
        output = self.conv(torch.cat([in1_up, inputs2_], 1))
        
        return output


def snr_torch(target, output):
    """
    Compute SNR between the target and the reconstructed images

    :param target:  numpy array of reference
    :param output:  numpy array we have produced
    :return: SNR in dB
    """
    if target.shape != output.shape:
        raise ValueError('There is something wrong with the dimensions!')
    return 10 * torch.log10(torch.sum(target ** 2) / torch.sum((target - output) ** 2))


def pcorr(target, output):
    mean_tar = torch.mean(target)
    mean_out = torch.mean(output)
    tar_dif = target - mean_tar
    out_dif = output - mean_out
    return torch.sum(tar_dif * out_dif) / (torch.sqrt(
        torch.sum(tar_dif ** 2)) * torch.sqrt(torch.sum(out_dif ** 2)))
