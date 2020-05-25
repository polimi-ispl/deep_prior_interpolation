from argparse import ArgumentParser


def parse_main_arguments():
    parser = ArgumentParser()
    
    # dataset parameter
    parser.add_argument('--imgdir', type=str, required=True, default='./data/',
                        help='The dir saving the processing data')
    parser.add_argument('--outdir', type=str, required=False, default='./data/',
                        help='The dir for saving the output data')
    parser.add_argument('--imgname', nargs='+', type=str, required=False, default=[],
                        help='The name of original images')
    parser.add_argument('--maskname', nargs='+', type=str, required=False, default=[],
                        help='The name of corrupted images')
    parser.add_argument('--gain', nargs='+', type=float, required=False, default=[2000],
                        help='normalize the input')
    parser.add_argument('--datadim', type=str, required=False, default='2d',
                        help='The dimension of the input data')
    parser.add_argument('--slice', type=str, required=False, default='XY',
                        help='The type of slice of 3D data, only usefull when datadim == 2.5d')
    parser.add_argument('--imgchannel', type=int, required=False, default=1,
                        help='The channel of processed image, when datadim==2.5d, representing the number \
                              of the slices of the 3D data.')

    # network design
    parser.add_argument('--gpu', type=int, required=False, default=-1,
                        help='GPU to use (lowest memory usage based)')
    parser.add_argument('--activation', type=str, default='LeakyReLU', required=False,
                        help='Activation function to be used in the convolution block [ReLU, Tanh, LeakyReLU]')
    parser.add_argument('--need_sigmoid', action='store_true',
                        help='Apply a sigmoid activation to the network output')
    parser.add_argument('--filters', nargs='+', type=int, required=False, default=[16, 32, 64, 128, 256],
                        help='Numbers of channels in every layer of encoder and decoder')
    parser.add_argument('--skip', nargs='+', type=int, required=False, default=[16, 32, 64, 128],
                        help='Number of channels for skip-connection')
    parser.add_argument('--inputdepth', type=int, required=False, default=64,
                        help='Depth of the input noise tensor')
    parser.add_argument('--upsample', type=str, required=False, default='nearest',
                        help='Upgoing deconvolution strategy for the network [nearest, bilinear, trilinear]')
    parser.add_argument('--inittype', type=str, required=False, default='xavier',
                        help='Initialization strategy for the network [default, normal, xavier, kaiming, orthogonal]')   
    parser.add_argument('--initgain', type=float, required=False, default=0.02,
                        help='Initialization scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--savemodel', action='store_true', help='If save the model')
    
    # training parameter
    parser.add_argument('--epochs', '-e', type=int, required=False, default=2001,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, required=False,
                        help='Learning Rate for Adam optimizer')
    parser.add_argument('--save_every', type=int, default=10000, required=False,
                        help='Number of epochs every which to save the results')
    parser.add_argument('--loss_max', type=float, default=30., required=False,
                        help='Maximum total loss for saving the image.')
    parser.add_argument('--param_noise', action='store_false',
                        help='Add normal noise to the parameters every epoch')
    parser.add_argument('--reg_noise_std', type=float, required=False, default=0.03,
                        help='Standard deviation of the normal noise to be added to the input every epoch')
    parser.add_argument('--noise_dist', type=str, default='n', required=False,
                        help='Input tensor type of noise [(n)ormal, (u)niform]')
    parser.add_argument('--noise_std', type=float, default=.1, required=False,
                        help='Input tensor noise standard deviation')

    return parser.parse_args()
