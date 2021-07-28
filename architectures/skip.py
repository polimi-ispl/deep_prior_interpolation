from torch import nn
from .base import Concat3D, Concat, get_activation, conv3d_mod, conv_mod


class Skip(nn.Module):
    def __init__(self,
                 num_input_channels=2,
                 num_output_channels=3,
                 num_channels_down=[16, 32, 64, 128, 128],
                 num_channels_up=[16, 32, 64, 128, 128],
                 num_channels_skip=[4, 4, 4, 4, 4],
                 filter_size_down=3,
                 filter_size_up=3,
                 filter_skip_size=1,
                 last_act_fun=None,
                 need_bias=True,
                 pad='zero',
                 upsample_mode='nearest',
                 downsample_mode='stride',
                 act_fun='LeakyReLU',
                 need1x1_up=True,
                 dropout=0.
                 ):
        
        super(Skip, self).__init__()
        
        self.model = _build_skip(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            num_channels_down=num_channels_down,
            num_channels_up=num_channels_up,
            num_channels_skip=num_channels_skip,
            filter_size_down=filter_size_down,
            filter_size_up=filter_size_up,
            filter_skip_size=filter_skip_size,
            last_act_fun=last_act_fun,
            need_bias=need_bias,
            pad=pad,
            upsample_mode=upsample_mode,
            downsample_mode=downsample_mode,
            act_fun=act_fun,
            need1x1_up=need1x1_up,
            dropout=dropout)
    
        self.parameters = self.model.parameters
        
    def forward(self, x):
        return self.model(x)
        

def _build_skip(num_input_channels=2,
                num_output_channels=3,
                num_channels_down=[16, 32, 64, 128, 128],
                num_channels_up=[16, 32, 64, 128, 128],
                num_channels_skip=[4, 4, 4, 4, 4],
                filter_size_down=3,
                filter_size_up=3,
                filter_skip_size=1,
                last_act_fun=None,
                need_bias=True,
                pad='zero',
                upsample_mode='nearest',
                downsample_mode='stride',
                act_fun='LeakyReLU',
                need1x1_up=True,
                dropout=0.):

    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    
    n_scales = len(num_channels_down)
    
    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    
    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales
    
    last_scale = n_scales - 1
    
    model = nn.Sequential()
    model_tmp = model
    
    input_depth = num_input_channels
    for i in range(n_scales):
        
        deeper = nn.Sequential()
        skip = nn.Sequential()
        
        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        model_tmp.add(nn.BatchNorm2d(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))
        
        if num_channels_skip[i] != 0:
            skip.add(conv_mod(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(nn.BatchNorm2d(num_channels_skip[i]))
            skip.add(get_activation(act_fun))
            skip.add(nn.Dropout2d(dropout))
            
        deeper.add(conv_mod(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                            downsample_mode=downsample_mode[i]))
        deeper.add(nn.BatchNorm2d(num_channels_down[i]))
        deeper.add(get_activation(act_fun))
        deeper.add(nn.Dropout2d(dropout))
        
        deeper.add(conv_mod(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(nn.BatchNorm2d(num_channels_down[i]))
        deeper.add(get_activation(act_fun))
        deeper.add(nn.Dropout2d(dropout))
        
        deeper_main = nn.Sequential()
        
        if i == last_scale:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]
        
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        
        model_tmp.add(
            conv_mod(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(nn.BatchNorm2d(num_channels_up[i]))
        model_tmp.add(get_activation(act_fun))
        model_tmp.add(nn.Dropout2d(dropout))
        
        if need1x1_up:
            model_tmp.add(conv_mod(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(nn.BatchNorm2d(num_channels_up[i]))
            model_tmp.add(get_activation(act_fun))
            model_tmp.add(nn.Dropout2d(dropout))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main
    
    model.add(conv_mod(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if isinstance(last_act_fun, str) and last_act_fun.lower() == 'none':
        last_act_fun = None
    if last_act_fun is not None:
        model.add(get_activation(last_act_fun))
    
    return model


def Skip3D(num_input_channels=2,
           num_output_channels=3,
           num_channels_down=[16, 32, 64, 128, 128],
           num_channels_up=[16, 32, 64, 128, 128],
           num_channels_skip=[4, 4, 4, 4, 4],
           filter_size_down=3,
           filter_size_up=3,
           filter_skip_size=1,
           last_act_fun=None,
           need_bias=True,
           pad='zero',
           upsample_mode='nearest',
           downsample_mode='stride',
           act_fun='LeakyReLU',
           need1x1_up=True,
           dropout=0.):

    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    
    n_scales = len(num_channels_down)
    
    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    
    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales
    
    last_scale = n_scales - 1
    
    model = nn.Sequential()
    model_tmp = model
    
    input_depth = num_input_channels
    for i in range(n_scales):
        
        deeper = nn.Sequential()
        skip = nn.Sequential()
        
        if num_channels_skip[i] != 0:
            model_tmp.add(Concat3D(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        model_tmp.add(nn.BatchNorm3d(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))
        
        if num_channels_skip[i] != 0:
            skip.add(conv3d_mod(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(nn.BatchNorm3d(num_channels_skip[i]))
            skip.add(get_activation(act_fun))
            skip.add(nn.Dropout3d(dropout))
        
        deeper.add(conv3d_mod(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                              downsample_mode=downsample_mode[i]))
        deeper.add(nn.BatchNorm3d(num_channels_down[i]))
        deeper.add(get_activation(act_fun))
        deeper.add(nn.Dropout3d(dropout))
        
        deeper.add(conv3d_mod(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(nn.BatchNorm3d(num_channels_down[i]))
        deeper.add(get_activation(act_fun))
        deeper.add(nn.Dropout3d(dropout))

        deeper_main = nn.Sequential()
        
        if i == last_scale:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]
        
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        
        model_tmp.add(
            conv3d_mod(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(nn.BatchNorm3d(num_channels_up[i]))
        model_tmp.add(get_activation(act_fun))
        model_tmp.add(nn.Dropout3d(dropout))

        if need1x1_up:
            model_tmp.add(conv3d_mod(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(nn.BatchNorm3d(num_channels_up[i]))
            model_tmp.add(get_activation(act_fun))
            model_tmp.add(nn.Dropout3d(dropout))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main
    
    model.add(conv3d_mod(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if isinstance(last_act_fun, str) and last_act_fun.lower() == 'none':
        last_act_fun = None
    if last_act_fun is not None:
        model.add(get_activation(last_act_fun))
    
    return model


__all__ = [
    "Skip",
    "Skip3D",
]
