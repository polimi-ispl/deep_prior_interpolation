from __future__ import print_function
import warnings
import os
import torch
import numpy as np
from time import time
from termcolor import colored

from parameter import parse_arguments
from architectures import UNet, MulResUnet, MulResUnet3D, AttMulResUnet2D, PartialConvUNet, PartialConv3DUNet
from data import get_patch, extract_patches
import utils as u

warnings.filterwarnings("ignore")
u.set_seed()


class Training:
    def __init__(self, args, outpath, dtype=torch.cuda.FloatTensor):
        self.args = args
        self.dtype = dtype
        self.outpath = outpath
        if args.loss == 'mse':
            self.lossfunc = torch.nn.MSELoss().type(self.dtype)
        else:
            self.lossfunc = torch.nn.L1Loss().type(self.dtype)
        self.elapsed = None
        self.iiter = 0
        self.saving_interval = 0
        self.loss_min = self.args.loss_max
        self.outchannel = args.imgchannel
        self.history = u.History()
        
        self.imgpath = None
        self.image_name = None
        self.img = None
        self.img_ = None
        self.mask = None
        self.mask_ = None
        self.out = None
        self.th = args.threshold
        self.step = args.update_step
        
        self.zfill = u.ten_digit(self.args.epochs)
        
        # build input tensors
        self.input_type = 'noise3d' if args.datadim == '3d' else 'noise'
        self.input_ = None
        self.input_old = None
        self.add_noise_ = None
        self.add_data_ = None
        self.add_data_weight = None
        self.input_list = []
        
        # build network
        self.net = None
        self.parameters = None
        self.num_params = None
    
    def build_input(self):
        # build a noise tensor
        data_shape = self.img.shape[:-1]
        self.input_ = u.get_noise(shape=(1, self.args.inputdepth) + data_shape,
                                  noise_type=self.args.noise_dist).type(self.dtype)
        self.input_ *= self.args.noise_std
        
        if self.args.filter_noise_with_data:
            dim = list(np.arange(len(data_shape)+2))
            dim = dim[:2] + dim[3:]
            traces_ = torch.mean(self.img_*self.mask_, dim=dim)
            traces_fft = torch.fft.fft(traces_)
        if self.args.filter_noise_with_wavelet:
                wav_fft = np.load(self.args.imgdir())
        if self.args.data_forgetting_factor != 0:
            # build decimated data tensor
            data_ = self.img_ * self.mask_
            # how many times we can repeat the data in order to fill the input depth?
            num_rep = int(np.ceil(self.args.inputdepth / self.args.imgchannel))
            # repeat data along the channel dim and crop to the input depth size
            data_ = data_.repeat([1, num_rep] + [1] * len(data_shape))[:, :self.args.inputdepth]
            # normalize data to noise std
            data_ *= torch.std(self.input_) / torch.std(data_)
            self.add_data_ = data_
            self.add_data_weight = np.logspace(0, -4, self.args.data_forgetting_factor)
        
        self.input_old = self.input_.detach().clone()
        self.add_noise_ = self.input_.detach().clone()
        print(colored('The input shape is %s' % str(tuple(self.input_.shape)), 'cyan'))
    
    def build_model(self, netpath: str = None):
        if self.args.datadim in ['2d', '2.5d']:
            if self.args.net == 'unet':
                self.net = UNet(
                    num_input_channels=self.args.inputdepth,
                    num_output_channels=self.outchannel,
                    filters=self.args.filters,
                    upsample_mode=self.args.upsample,  # default is bilinear
                    need_sigmoid=self.args.need_sigmoid,
                    need_bias=True,
                    activation=self.args.activation  #
                )
            elif self.args.net == 'attmultiunet':
                self.net = AttMulResUnet2D(
                    num_input_channels=self.args.inputdepth,
                    num_output_channels=self.outchannel,
                    num_channels_down=self.args.filters,
                    upsample_mode=self.args.upsample,  # default is bilinear
                    need_sigmoid=self.args.need_sigmoid,
                    need_bias=True,
                    act_fun=self.args.activation  # default is LeakyReLU).type(self.dtype)
                )
            elif self.args.net == 'part':
                self.net = PartialConvUNet(self.args.inputdepth, self.outchannel)
            else:
                self.net = MulResUnet(
                    num_input_channels=self.args.inputdepth,
                    num_output_channels=self.outchannel,
                    num_channels_down=self.args.filters,
                    num_channels_up=self.args.filters,
                    num_channels_skip=self.args.skip,
                    upsample_mode=self.args.upsample,  # default is bilinear
                    need_sigmoid=self.args.need_sigmoid,
                    need_bias=True,
                    act_fun=self.args.activation  # default is LeakyReLU
                )
        else:
            if self.args.net == 'part':
                self.net = PartialConv3DUNet(self.args.inputdepth, self.outchannel)
            elif self.args.net == 'load':
                self.net = MulResUnet3D(
                    num_input_channels=self.args.inputdepth,
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
            else:
                self.net = MulResUnet3D(
                    num_input_channels=self.args.inputdepth,
                    num_output_channels=self.outchannel,
                    num_channels_down=self.args.filters,
                    num_channels_up=self.args.filters,
                    num_channels_skip=self.args.skip,
                    upsample_mode=self.args.upsample,  # default is bilinear
                    need_sigmoid=self.args.need_sigmoid,
                    need_bias=True,
                    act_fun=self.args.activation  # default is LeakyReLU).type(self.dtype)
                )
        
        self.net = self.net.type(self.dtype)
        
        if self.args.net != 'load':
            u.init_weights(self.net, self.args.inittype, self.args.initgain)
        self.parameters = u.get_params('net', self.net, self.input_)
        self.num_params = sum(np.prod(list(p.size())) for p in self.net.parameters())
    
    def load_data(self, data):
        """
        Load the full patch and mask
        Parameters:
            data -- the dictionary include the attribute of 'image', 'mask', 'name', as created by "data.py" file.
        """
        self.image_name = data['name']  # here the name is set as the name of input patch.
        self.img = data['image']
        self.mask = data['mask']
        
        if self.mask.shape != self.img.shape:
            raise ValueError('The loaded mask shape has to be', self.img.shape)
        
        sha = tuple(range(len(self.img.shape)))
        re_sha = sha[-1:] + sha[:-1]
        
        self.img_ = u.np_to_torch(np.transpose(self.img, re_sha)[np.newaxis]).type(self.dtype)
        self.mask_ = u.np_to_torch(np.transpose(self.mask, re_sha)[np.newaxis]).type(self.dtype)
        
        self.mask_update = u.MaskUpdate(self.mask_, self.th, self.step)
        
        # compute std for skipping all-zeros patches
        input_std = torch.std(self.img_ * self.mask_).item()
        return input_std
    
    def optimization_loop(self):
        # Adding normal noise to the learned parameters.
        if self.args.param_noise:
            for n in [x for x in self.net.parameters() if len(x.size()) in [4, 5]]:
                n = n + n.detach().clone().normal_() * n.std() * 0.02
        
        # Adding normal noise to the input noise.
        input_ = self.input_old
        if self.args.reg_noise_std > 0:
            input_ = self.input_old + (self.add_noise_.normal_() * self.args.reg_noise_std)
        
        # adding data to the input noise
        if self.iiter < self.args.data_forgetting_factor:
            input_ += self.add_data_weight[self.iiter] * self.add_data_
            self.input_list.append(u.torch_to_np(input_[0, 0]))
        
        # compute output
        output_ = self.net(input_)
        
        if self.iiter % self.step == 0:
            self.output_old = output_.clone().detach()
        
        # compute the new mask and new related input
        mask_ = self.mask_update.update(self.iiter)
        image_ = self.img_ * self.mask_ + (mask_ - self.mask_) * self.output_old
        
        # compute the loss function
        total_loss = self.lossfunc(output_ * mask_, image_)
        total_loss.backward()
        
        # save loss and metrics, and print log
        l = total_loss.item()
        s = u.snr(output=output_ * (1 - self.mask_) + self.img_ * self.mask_, target=self.img_).item()
        p = u.pcorr(output=output_ * (1 - self.mask_), target=self.img_ * (1 - self.mask_)).item()
        self.history.append((l, s, p))
        
        msg = "Iter %s, Loss = %.2e, SNR = %+.2f dB, PCORR = %+.2f %%" \
              % (str(self.iiter + 1).zfill(self.zfill), l, s, p * 100)
        
        print(colored(msg, 'yellow'), '\r', end='')
        
        # The early stop, save if the Loss is decreasing (above a threshold)
        if self.loss_min > self.history.loss[-1]:
            self.loss_min = self.history.loss[-1]
            if len(output_.shape) <= 4:
                self.out = u.torch_to_np(output_).transpose(1, 2, 0)
            else:
                self.out = u.torch_to_np(output_).squeeze()
        
        # saving the intermediate output. if don't want to save anything, set save_every larger than epoches
        if self.iiter in list(range(0, 200, int(self.args.save_every))) and self.iiter != 0:
            if len(output_.shape) <= 4:
                out_img = u.torch_to_np(output_).transpose(1, 2, 0)
            else:
                out_img = u.torch_to_np(output_).squeeze()
            np.save(os.path.join(self.outpath, self.image_name + '_' + str(self.iiter) + '.npy'),
                    out_img)
        
        self.iiter += 1
        
        return total_loss
    
    def optimize(self):
        """
        Train the network. For each iteration, call the optimization loop function.
        """
        print(colored('starting optimization with ADAM...', 'cyan'))
        optimizer = torch.optim.Adam(self.parameters, lr=self.args.lr)
        start = time()
        for j in range(self.args.epochs):
            optimizer.zero_grad()
            self.optimization_loop()
            optimizer.step()
        
        self.elapsed = time() - start
        print(colored(u.sec2time(self.elapsed), 'yellow'))
    
    def save_result(self):
        """
        Save the results, the model (if asked) and some info to disk in a .npy file.
        """
        np.save(os.path.join(self.outpath, self.image_name + '_run.npy'), {
            'device' : u.get_gpu_name(int(os.environ["CUDA_VISIBLE_DEVICES"])),
            'elapsed': u.sec2time(self.elapsed),
            'outpath': self.outpath,
            'history': self.history,
            # 'args'   : self.args,
            'mask'   : self.mask,
            'image'  : self.img,
            'output' : self.out,
            'noise'  : self.input_list,
        })
        
        # save the model
        if self.args.savemodel:
            torch.save(self.net.state_dict(),
                       os.path.join(self.outpath, self.image_name + '_model.pth'))
    
    def clean(self):
        """
        Clean the trainer for a new patch.
        """
        self.iiter = 0
        self.saving_interval = 0
        print(colored('Finished patch %s' % self.image_name, 'yellow'))
        torch.cuda.empty_cache()
        self.loss_min = self.args.loss_max
        self.history = u.History()


def main() -> None:
    args = parse_arguments()
    
    u.set_gpu(args.gpu)
    
    # create output folder and save arguments in a .txt file
    outpath = os.path.join('./results/', args.outdir if args.outdir is not None else u.random_code())
    os.makedirs(outpath, exist_ok=True)
    print(colored('Saving to %s' % outpath, 'yellow'))
    u.write_args(os.path.join(outpath, 'args.txt'), args)
    
    # get a list of patches organized as dictionaries with image, mask and name fields
    if args.use_pe:
        patches = extract_patches(args)
    else:
        patches = get_patch(args)
    
    print(colored('Processing %d patches' % len(patches), 'yellow'))
    
    # instantiate a trainer
    T = Training(args, outpath)
    
    # interpolation
    for i, patch in enumerate(patches):
        
        print(colored('\nThe image shape is %s' % str(patch['image'].shape), 'cyan'))
        
        std = T.load_data(patch)
        print(colored('the std of input image is %.2e, ' % std, 'cyan'), end="")
        
        if np.isclose(std, 0., atol=1e-12):  # all the data are corrupted
            print(colored('skipping...', 'cyan'))
            T.out = T.img * T.mask
            T.elapsed = 0.
        else:
            # TODO add the transfer learning option
            if i == 0 or not args.start_from_prev:
                T.build_model()
            T.build_input()
            T.optimize()
        
        T.save_result()
        T.clean()
    
    print(colored('Interpolation done! Saved to %s' % outpath, 'yellow'))


if __name__ == '__main__':
    main()
