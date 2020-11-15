from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import os
import torch
import numpy as np
from collections import namedtuple
from time import time
from termcolor import colored
import json

from parameter import parse_arguments
from architectures import MulResUnet, MulResUnet3D, AttMulResUnet2D, PartialConvUNet, PartialConv3DUNet
from data import get_patch
import utils as u


dtype = torch.cuda.FloatTensor
u.set_seed()
# this is defined here because of pickle
History = namedtuple("History", ['loss', 'snr', 'pcorr'])


class Training:
    def __init__(self, args, outpath, dtype=dtype):
        self.args = args
        self.dtype = dtype
        self.outpath = outpath
        # MSELoss()
        self.lossfunc = torch.nn.L1Loss().type(self.dtype)
        self.elapsed = None
        self.iiter = 0
        self.saving_interval = 0
        self.loss_min = self.args.loss_max
        self.outchannel = args.imgchannel
        self.history = History([], [], [])
        
        self.imgpath = None
        self.image_name = None
        self.img = None
        self.img_tensor = None
        self.mask = None
        self.mask_tensor = None
        self.out_img = None
        self.threshold = args.threshold
        self.step = args.update_step
        
        # build input tensors
        self.input_type = 'noise3d' if args.datadim == '3d' else 'noise'
        self.input_tensor = None
        self.input_tensor_old = None
        self.additional_noise_tensor = None
        
        # build network
        self.net = None
        self.parameters = None
        self.num_params = None
    
    def build_input(self, img_shape):
        # add the input_type
        self.input_tensor = u.get_noise(self.args.inputdepth, img_shape[:-1], self.input_type,
                                        noise_type=self.args.noise_dist, var=self.args.noise_std).type(dtype)
        self.input_tensor_old = self.input_tensor.detach().clone()
        self.additional_noise_tensor = self.input_tensor.detach().clone()
        print(colored('The shape of input noise is %s.\n' % str(self.input_tensor.shape), 'yellow'))
    
    def build_model(self, netpath=None):
        if self.args.datadim in ['2d', '2.5d']:
            if self.args.net == 'multiunet':
                self.net = MulResUnet(num_input_channels=self.args.inputdepth,
                                      num_output_channels=self.outchannel,
                                      num_channels_down=self.args.filters,
                                      num_channels_up=self.args.filters,
                                      num_channels_skip=self.args.skip,
                                      upsample_mode=self.args.upsample,  # default is bilinear
                                      need_sigmoid=self.args.need_sigmoid,
                                      need_bias=True,
                                      act_fun=self.args.activation  # default is LeakyReLU).type(self.dtype)
                                      ).type(self.dtype)
            elif self.args.net == 'attmultiunet':
                self.net = AttMulResUnet2D(num_input_channels=self.args.inputdepth,
                                           num_output_channels=self.outchannel,
                                           num_channels_down=self.args.filters,
                                           upsample_mode=self.args.upsample,  # default is bilinear
                                           need_sigmoid=self.args.need_sigmoid,
                                           need_bias=True,
                                           act_fun=self.args.activation  # default is LeakyReLU).type(self.dtype)
                                           ).type(self.dtype)
            elif self.args.net == 'part':
                self.net = PartialConvUNet(self.args.inputdepth, self.outchannel).type(self.dtype)
        
        else:
            if self.args.net == 'part':
                self.net = PartialConv3DUNet(self.args.inputdepth, self.outchannel).type(self.dtype)
            elif self.args.net == 'load':
                self.net = MulResUnet3D(num_input_channels=self.args.inputdepth,
                                        num_output_channels=self.outchannel,
                                        num_channels_down=self.args.filters,
                                        num_channels_up=self.args.filters,
                                        num_channels_skip=self.args.skip,
                                        upsample_mode=self.args.upsample,  # default is bilinear
                                        need_sigmoid=self.args.need_sigmoid,
                                        need_bias=True,
                                        act_fun=self.args.activation  # default is LeakyReLU).type(self.dtype)
                                        )
                self.net.load_state_dict(torch.load(netpath))
                self.net = self.net.type(self.dtype)
            else:
                self.net = MulResUnet3D(num_input_channels=self.args.inputdepth,
                                        num_output_channels=self.outchannel,
                                        num_channels_down=self.args.filters,
                                        num_channels_up=self.args.filters,
                                        num_channels_skip=self.args.skip,
                                        upsample_mode=self.args.upsample,  # default is bilinear
                                        need_sigmoid=self.args.need_sigmoid,
                                        need_bias=True,
                                        act_fun=self.args.activation  # default is LeakyReLU).type(self.dtype)
                                        ).type(self.dtype)
        if self.args.net != 'load':
            u.init_weights(self.net, self.args.inittype, self.args.initgain)
        self.parameters = u.get_params('net', self.net, self.input_tensor)
        self.num_params = sum(np.prod(list(p.size())) for p in self.net.parameters())
    
    def load_data(self, data):
        """ Load the corrupted image and mask 
            Parameters:
                data -- the dictionary include the attribute of 'img', 'mask', 'name', 
                        the output of get_patch function.
        """
        self.image_name = data['name'].split('.')[0]  # here the name is set as the name of input image.
        self.img = data['img']
        self.mask = data['mask']
        
        if self.mask.shape != self.img.shape:
            raise ValueError('The loaded mask shape has to be', self.img.shape)
        
        sha = tuple(range(len(self.img.shape)))
        re_sha = sha[-1:] + sha[:-1]
        
        self.img_tensor = u.np_to_torch(np.transpose(self.img, re_sha)[np.newaxis]).type(self.dtype)
        self.mask_tensor = u.np_to_torch(np.transpose(self.mask, re_sha)[np.newaxis]).type(self.dtype)
        
        self.mask_update = u.MaskUpdate(self.mask_tensor, self.threshold, self.step)
        
        input_std = torch.std(self.img_tensor * self.mask_tensor).item()
        return input_std
    
    def optimization_loop(self):
        # Adding normal noise to the learned parameters.
        if self.args.param_noise:
            for n in [x for x in self.net.parameters() if len(x.size()) in [4, 5]]:
                n = n + n.detach().clone().normal_() * n.std() * 0.02
        
        # Adding normal noise to the input noise.
        input_tensor = self.input_tensor_old
        if self.args.reg_noise_std > 0:
            input_tensor = self.input_tensor_old + (self.additional_noise_tensor.normal_() * self.args.reg_noise_std)
        
        # mask_tensor = self.mask_tensor.repeat((1, self.args.inputdepth, 1, 1, 1))
        output_tensor = self.net(input_tensor)
        
        mask_tensor = self.mask_update.update(self.iiter)
        if self.iiter % self.step == 0:
            self.old_output = output_tensor.clone().detach()
        
        image_tensor = self.img_tensor * self.mask_tensor + (mask_tensor - self.mask_tensor) * self.old_output
        
        total_loss = self.lossfunc(output_tensor * mask_tensor, image_tensor)
        
        total_loss.backward()
        
        self.history.loss.append(total_loss.item())
        self.history.snr.append(
            u.snr(self.img_tensor, output_tensor * (1 - self.mask_tensor) + self.img_tensor * self.mask_tensor).item())
        self.history.pcorr.append(
            u.pcorr(self.img_tensor * (1 - self.mask_tensor), output_tensor * (1 - self.mask_tensor)).item())
        
        msg = "Iter %s, Loss = %.2e, SNR = %+.2f dB, PCORR = %+.2e" \
              % (str(self.iiter + 1).zfill(u.ten_digit(self.args.epochs)),
                 self.history.loss[-1],
                 self.history.snr[-1],
                 self.history.pcorr[-1])
        
        print(colored(msg, 'yellow'), '\r', end='')
        
        # The early stop, save if the Loss is decreasing (above a threshold)
        
        if self.loss_min > self.history.loss[-1]:
            self.loss_min = self.history.loss[-1]
            if len(output_tensor.shape) <= 4:
                self.out_img = u.torch_to_np(output_tensor).transpose(1, 2, 0)
            else:
                self.out_img = u.torch_to_np(output_tensor).squeeze()
        # saving the intermediate output. if don't want to save anything, set save_every larger than epoches
        if self.iiter in list(range(0, 200, int(self.args.save_every))) and self.iiter != 0:
            if len(output_tensor.shape) <= 4:
                out_img = u.torch_to_np(output_tensor).transpose(1, 2, 0)
            else:
                out_img = u.torch_to_np(output_tensor).squeeze()
            np.save(os.path.join(self.outpath, self.image_name + '_' + str(self.iiter) + '.npy'),
                    out_img)
        
        self.iiter += 1
        # self.saving_interval += 1
        
        return total_loss
    
    def optimize(self):
        """
            Train the network.
        """
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(self.parameters, lr=self.args.lr)
        start = time()
        for j in range(self.args.epochs):
            optimizer.zero_grad()
            self.optimization_loop()
            optimizer.step()
        
        self.elapsed = time() - start
        print(colored(u.sec2time(self.elapsed), 'yellow'))
    
    def save_result(self):
        mydict = {
            'device'      : os.environ["CUDA_VISIBLE_DEVICES"],
            'elapsed time': u.sec2time(self.elapsed),
            'run_code'    : self.outpath[-6:],
            'history'     : self.history,
            'args'        : self.args,
            'mask'        : self.mask,
            'image'       : self.img,
            'output'      : self.out_img
        }
        np.save(os.path.join(self.outpath, self.image_name + '_run.npy'), mydict)
        # save the model
        if self.args.savemodel:
            torch.save(self.net.state_dict(),
                       os.path.join(self.outpath, self.image_name + '_latest_net.pth'))
    
    def clean(self):
        self.iiter = 0
        self.saving_interval = 0
        print(colored('The image -%s has finished' % self.image_name, 'green'))
        torch.cuda.empty_cache()
        self.loss_min = self.args.loss_max


def main() -> None:
    args = parse_arguments()
    
    u.set_gpu(args.gpu)
    
    # create output folder
    dir_list = list(filter(None, args.imgdir.split('/')))
    outpath = os.path.join(args.outdir, u.random_code())
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    print(colored('Saving to %s' % outpath, 'yellow'))
    with open(os.path.join(outpath, 'args.txt'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    
    T = Training(args, outpath, dtype=dtype)
    
    patches_list = get_patch(args)
    
    for i, patch in enumerate(patches_list):
        imgshape = patch['img'].shape
        print(colored('The image shape is %s ' % (imgshape,), 'green'))
        
        T.build_input(imgshape)
        T.build_model()
        std = T.load_data(patch)
        print('the std of input image is %f' % std)
        if np.isclose(std, 0.):  # all the data are corrupted
            print('skipping...')
            T.out_img = T.img * T.mask
            T.elapsed = 0.
        else:
            T.optimize()
        T.save_result()
        T.clean()
    print(colored('Interpolation done! Saved to %s' % outpath, 'yellow'))


if __name__ == '__main__':
    main()
