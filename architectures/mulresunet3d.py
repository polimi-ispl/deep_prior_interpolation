import torch
from torch import nn
import numpy as np
from .base import act, Symmetry

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
    "MulResUnet3D",
]
