from __future__ import print_function
import warnings
import os
import torch
import numpy as np
from time import time
from termcolor import colored

from parameter import parse_arguments, net_args_are_same
from architectures import get_net
import utils as u
from data import extract_patches

warnings.filterwarnings("ignore")
u.set_seed()


class Interpolator:
    def __init__(self, args, outpath):
        
        self.args = args
        self.dtype = torch.FloatTensor if args.gpu is None else torch.cuda.FloatTensor
        self.outpath = outpath
        if args.loss == 'mse':
            self.loss_fn = torch.nn.MSELoss().type(self.dtype)
        else:
            self.loss_fn = torch.nn.L1Loss().type(self.dtype)
        self.elapsed = None
        self.iiter = 0
        self.iter_to_be_saved = list(range(0, self.args.epochs, int(self.args.save_every))) \
            if self.args.save_every is not None else [0]
        self.loss_min = None
        self.outchannel = args.imgchannel
        self.history = u.History(self.args.epochs)
        
        self.imgpath = None
        self.image_name = None
        self.img = None
        self.img_ = None
        self.mask = None
        self.mask_ = None
        self.out_best = None
        self.out_old = None
        self.zfill = u.ten_digit(self.args.epochs)
        
        # build input tensors
        self.input_type = 'noise3d' if args.datadim == '3d' else 'noise'
        self.input_ = None
        self.add_data_ = None
        self.add_data_weight = None
        self.input_list = []
        
        # build network
        self.net = None
        self.parameters = None
        self.num_params = None
        self.optimizer = None
        self.netdir = None
    
    def build_input(self):
        # build a noise tensor
        data_shape = self.img.shape[:-1]
        self.input_ = u.get_noise(shape=(1, self.args.inputdepth) + data_shape,
                                  noise_type=self.args.noise_dist).type(self.dtype)
        self.input_ *= self.args.noise_std
        
        if self.args.filter_noise_with_wavelet:
            W = u.ConvolveKernel_1d(
                kernel=np.load(os.path.join(self.args.imgdir, 'wavelet.npy')),
                ndim=self.input_.ndim - 2,
                dtype=self.dtype,
            )
            self.input_ = W(self.input_)
        
        if self.args.lowpass_fs and self.args.lowpass_fc:
            print(colored("filtering the input tensor with a low pass Butterworth...", "cyan"))
            # low pass filter input noise tensor with a 4th order butterworth
            LPF = u.LowPassButterworth(fc=self.args.lowpass_fc,
                                       ndim=self.input_.ndim-2,
                                       fs=self.args.lowpass_fs,
                                       ntaps=self.args.lowpass_ntaps,
                                       order=4,
                                       nfft=2**u.nextpow2(self.input_.shape[2]),
                                       dtype=self.dtype)
            self.input_ = LPF(self.input_)
        
        if self.args.data_forgetting_factor != 0:
            # build decimated data tensor
            data_ = self.img_ * self.mask_
            # how many times we can repeat the data in order to fill the input depth?
            num_rep = int(np.ceil(self.input_.shape[1] / data_.shape[1]))
            # num_rep = self.args.inputdepth if self.args.datadim == "3d" else int(np.ceil(self.args.inputdepth / self.args.imgchannel))
            # repeat data along the channel dim and crop to the input depth size
            data_ = data_.repeat([1, num_rep] + [1] * len(data_shape))[:, :self.args.inputdepth]
            # normalize data to noise std
            data_ *= torch.std(self.input_) / torch.std(data_)
            self.add_data_ = data_
            self.add_data_weight = np.logspace(0, -4, self.args.data_forgetting_factor)
        
        print(colored('The input shape is %s' % str(tuple(self.input_.shape)), 'cyan'))
    
    def build_model(self, netpath: str = None):
        if self.outchannel is None:
            self.outchannel = self.img_.shape[1]
        
        if len(self.args.netdir) != 0:
            _args = u.read_args(os.path.join('./results', *netpath.split('/')[:-1], "args.txt"))
            assert net_args_are_same(self.args, _args)
            self.net = get_net(_args, self.outchannel).type(self.dtype)
            self.net.load_state_dict(torch.load(os.path.join('./results', netpath)))
            print(colored("Network loaded from %s" % os.path.join('./results', netpath), "cyan"))
        else:
            self.net = get_net(self.args, self.outchannel).type(self.dtype)
            u.init_weights(self.net, self.args.inittype, self.args.initgain)
            
        # self.parameters = u.get_params('net', self.net, self.input_)
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
        
        sha = tuple(range(self.img.ndim))
        re_sha = sha[-1:] + sha[:-1]
        
        self.img_ = u.np_to_torch(np.transpose(self.img, re_sha), bc_add=False).unsqueeze(0).type(self.dtype)
        self.mask_ = u.np_to_torch(np.transpose(self.mask, re_sha), bc_add=False).unsqueeze(0).type(self.dtype)
        
        # compute std on coarse data for skipping all-zeros patches
        input_std = torch.std(self.img_ * self.mask_).item()
        return input_std
    
    def optimization_loop(self):
        # adding normal noise to the learned parameters
        if self.args.param_noise:
            for n in [x for x in self.net.parameters() if len(x.size()) in [4, 5]]:
                n = n + n.detach().clone().normal_() * n.std() * 0.02
        
        # adding normal noise to the input tensor
        input_ = self.input_.detach().clone()
        if self.args.reg_noise_std > 0:
            input_ += self.args.reg_noise_std * input_.clone().normal_()

        # adding data to the input noise
        if self.iiter < self.args.data_forgetting_factor:
            input_ += self.add_data_weight[self.iiter] * self.add_data_
            self.input_list.append(u.torch_to_np(input_, True))
        
        # compute output
        out_ = self.net(input_)
        
        # compute the loss function
        total_loss = self.loss_fn(out_ * self.mask_, self.img_ * self.mask_)
        total_loss.backward()
        
        # save loss and metrics, and print log
        l = total_loss.item()
        s = u.snr(output=out_, target=self.img_).item()
        p = u.pcorr(output=out_, target=self.img_).item()
        self.history.append((l, s, p))
        self.history.lr.append(self.optimizer.param_groups[0]['lr'])
        print(colored(self.history.log_message(self.iiter), 'yellow'), '\r', end='')
        
        # save the output if the loss is decreasing
        if self.iiter == 0:
            self.loss_min = self.history.loss[-1]
            self.out_best = u.torch_to_np(out_, True) if out_.ndim > 4 else \
                u.torch_to_np(out_, False)[0].transpose((1, 2, 0))
        elif self.history.loss[-1] <= self.loss_min:
            self.loss_min = self.history.loss[-1]
            self.out_best = u.torch_to_np(out_, True) if out_.ndim > 4 else \
                u.torch_to_np(out_, False)[0].transpose((1, 2, 0))
        else:
            pass
        
        # saving intermediate outputs
        if self.iiter in self.iter_to_be_saved and self.iiter != 0:
            out_img = u.torch_to_np(out_, True) if out_.ndim > 4 else u.torch_to_np(out_, False)[0].transpose((1, 2, 0))
            np.save(os.path.join(self.outpath,
                                 self.image_name.split('.')[0] + '_output%s.npy' % str(self.iiter).zfill(self.zfill)),
                    out_img)
        
        self.iiter += 1
        
        return total_loss
    
    def optimize(self):
        """
        Train the network. For each iteration, call the optimization loop function.
        """
        print(colored('starting optimization with ADAM...', 'cyan'))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                               factor=self.args.lr_factor,
                                                               threshold=self.args.lr_thresh,
                                                               patience=self.args.lr_patience)
        # stop after no improvements greater than a certain percentage of the previous loss
        stopper = u.EarlyStopping(patience=self.args.earlystop_patience,
                                  min_delta=self.args.earlystop_min_delta,
                                  percentage=True)
        start = time()
        for j in range(self.args.epochs):
            self.optimizer.zero_grad()
            loss = self.optimization_loop()
            self.optimizer.step()
            if self.args.reduce_lr:
                scheduler.step(loss)
            if stopper.step(loss):  # stopper is computed on loss, as we don't have any validation metrics
                break
        
        self.elapsed = time() - start
        print(colored(u.sec2time(self.elapsed), 'yellow'))
    
    def save_result(self):
        """
        Save the results, the model (if asked) and some info to disk in a .npy file.
        """
        np.save(os.path.join(self.outpath, self.image_name + '_run.npy'), {
            'device' : u.get_gpu_name(),
            'elapsed': u.sec2time(self.elapsed),
            'outpath': self.outpath,
            'history': self.history,
            'mask'   : self.mask,
            'image'  : self.img,
            'output' : self.out_best,
            'noise'  : self.input_list,
        })
        
        # save the model
        if self.args.savemodel:
            torch.save(self.net.state_dict(),
                       os.path.join(self.outpath, self.image_name + '_model.pth'))
    
    def clean(self):
        """
        Clean the trainer for a new patch.
        Don't touch the model, as it depends on transfer learning options.
        """
        self.iiter = 0
        print(colored('Finished patch %s' % self.image_name, 'yellow'))
        torch.cuda.empty_cache()
        self.loss_min = None
        self.history = u.History(self.args.epochs)


def main() -> None:
    args = parse_arguments()
    
    u.set_gpu(args.gpu)
    
    # create output folder and save arguments in a .txt file
    outpath = os.path.join('./results/', args.outdir if args.outdir is not None else u.random_code())
    os.makedirs(outpath, exist_ok=True)
    print(colored('Saving to %s' % outpath, 'yellow'))
    u.write_args(os.path.join(outpath, 'args.txt'), args)
    
    # get a list of patches organized as dictionaries with image, mask and name fields
    patches = extract_patches(args)
    
    print(colored('Processing %d patches' % len(patches), 'yellow'))
    
    # instantiate an interpolator
    T = Interpolator(args, outpath)
    
    # interpolation
    for i, patch in enumerate(patches):
        
        print(colored('\nThe data shape is %s, ' % str(patch['image'].shape), 'cyan'), end="")
        
        std = T.load_data(patch)
        print(colored('the std of coarse data is %.2e' % std, 'cyan'))
        
        if np.isclose(std, 0., atol=1e-12):  # all the data are corrupted
            print(colored('skipping...', 'cyan'))
            T.out_best = T.img * T.mask
            T.elapsed = 0.
        else:
            if T.net is None or not args.start_from_prev:
                if len(args.netdir) != 0:
                    T.build_model(netpath=args.netdir[i])
                else:
                    T.build_model()
            T.build_input()
            T.optimize()
        
        T.save_result()
        T.clean()
    
    print(colored('Interpolation done! Saved to %s' % outpath, 'yellow'))


if __name__ == '__main__':
    main()
