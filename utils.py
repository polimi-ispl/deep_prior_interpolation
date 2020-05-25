import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from GPUtil import getFirstAvailable, getGPUs
from termcolor import colored
import os
import random
import string


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal | default
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'xavier' for the network.
    """
    from termcolor import colored
    
    def init_func(m):  # define a initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 10.0, init_gain * 10)
            init.constant_(m.bias.data, 0.0)
    
    if init_type != 'default':
        net.apply(init_func)
        print(colored('initialize network with %s' % init_type, 'red'))


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(input_depth, spatial_size, method='noise', noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the input noise tensor.
        method: `noise` for 2d convolutional network; `noise3d` for 3D convolutional network.
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'noise3d':
        assert len(spatial_size) == 3
        shape = [1, input_depth, spatial_size[0], spatial_size[1], spatial_size[2]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var
    else:
        assert False
    return net_input


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def ten_digit(number):
    return int(np.floor(np.log10(number)) + 1)


def sec2time(seconds):
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // 3600
    timestamp = '%dh:%dm:%ds' % (h, m, s)
    return timestamp


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
        
        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
    
    return params


def set_gpu(id=-1):
    """
    Set GPU device or select the one with the lowest memory usage (None for
    CPU-only)
    """
    if id is None:
        # CPU only
        print(colored('GPU not selected', 'yellow'))
    else:
        # -1 for automatic choice
        device = id if id is not -1 else getFirstAvailable(order='memory')[0]
        try:
            name = getGPUs()[device].name
        except IndexError:
            print('The selected GPU does not exist. Switching to the most '
                  'available one.')
            device = getFirstAvailable(order='memory')[0]
            name = getGPUs()[device].name
        
        print(colored('GPU selected: %d - %s' % (device, name), 'yellow'))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)


def random_code(n=6):
    return ''.join([random.choice(string.ascii_letters + string.digits)
                    for _ in range(int(n))])


def set_seed(seed=0):
    """
        Set the seed of random.
    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
