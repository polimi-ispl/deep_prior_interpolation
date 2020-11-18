from argparse import ArgumentParser, Namespace


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    # dataset parameter
    parser.add_argument('--imgdir', type=str, required=True, default='./datasets/',
                        help='Directory containing the processed data')
    parser.add_argument('--outdir', type=str, required=False,
                        help='Subfolder in ./results/ for saving.')
    parser.add_argument('--imgname', type=str, help='The name of original images')
    parser.add_argument('--maskname', type=str, help='The name of corrupted images')
    parser.add_argument('--gain', type=float, required=False, default=2e3,
                        help='gain for the input')
    parser.add_argument('--datadim', type=str, required=False, default='2d', choices=['2d', '2.5d', '3d'],
                        help='The dimensionality of the data')
    parser.add_argument('--slice', type=str, required=False, default='XY', choices=['XT', 'YT', 'XY'],
                        help='The type of slice of 3D data, only usefull when datadim=2.5d')
    parser.add_argument('--imgchannel', type=int, required=False, default=1,
                        help='Number of 2.5d patches to be stacked in the channel dimension.')
    parser.add_argument('--adirandel', type=float, required=False, default=0.,
                        help='The percent of addictive random deleting samples')
    parser.add_argument('--padwidth', type=int, required=False, default=0,
                        help='The padding width to the process data using edge mode')
    parser.add_argument('--patch_shape', nargs='+', type=int, required=False,
                        help="Patch shape to be processed (it can handle 2D, 2.5D, 3D)")
    parser.add_argument('--patch_stride', nargs='+', type=int, required=False,
                        help="Patch stride for the extraction (it can handle 2D, 2.5D, 3D)")

    # network design
    parser.add_argument('--net', type=str, required=False, default='multiunet',
                        choices=['multiunet', 'attmultiunet', 'part', 'multiunet3d', 'load'],
                        help='The architecture')
    parser.add_argument('--gpu', type=int, required=False, default=-1,
                        help='GPU to use (default lowest memory usage)')
    parser.add_argument('--activation', type=str, default='LeakyReLU', required=False,
                        choices=['LeakyReLU', 'ReLU', 'Tanh'],
                        help='Activation function to be used in the convolution block')
    parser.add_argument('--need_sigmoid', action='store_true', default=False,
                        help='Apply a sigmoid activation to the network output')
    parser.add_argument('--filters', nargs='+', type=int, required=False, default=[16, 32, 64, 128, 256],
                        help='Numbers of channels in every layer of encoder and decoder')
    parser.add_argument('--skip', nargs='+', type=int, required=False, default=[16, 32, 64, 128],
                        help='Number of channels for skip-connection')
    parser.add_argument('--inputdepth', type=int, required=False, default=64,
                        help='Depth of the input noise tensor')
    parser.add_argument('--upsample', type=str, required=False, default='nearest',
                        choices=['nearest', 'bilinear', 'trilinear'],
                        help="Network's upgoing deconvolution strategy")
    parser.add_argument('--inittype', type=str, required=False, default='xavier',
                        choices=['xavier', 'normal', 'default', 'kaiming', 'orthogonal'],
                        help='Initialization strategy for the network weights')
    parser.add_argument('--initgain', type=float, required=False, default=0.02,
                        help='Initialization scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--savemodel', action='store_true', default=False,
                        help='Save the optimized model to disk')
    parser.add_argument('--netdir', type=str, required=False, default='',
                        help='Path for saving the optimized model')
    # training parameter
    parser.add_argument('--epochs', '-e', type=int, required=False, default=2001,
                        help='Number of optimization iterations')
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
    parser.add_argument('--noise_dist', type=str, default='u', required=False, choices=['u', 'n'],
                        help='Type of noise for the input tensor [normal(n), uniform(u)]')
    parser.add_argument('--noise_std', type=float, default=.1, required=False,
                        help='Standard deviation of the noise for the input tensor')
    parser.add_argument('--gain_sym', type=float, default=0.01, required=False,
                        help='The weight of the symmetry loss')
    parser.add_argument('--threshold', type=int, default=10000, required=False,
                        help='Update mask when iteration larger than threshold')
    parser.add_argument('--update_step', type=int, default=200, required=False,
                        help='Update mask step')
    
    args = parser.parse_args()
    assert len(args.imgname) == len(args.maskname), "Images and Masks have to be the same number"
    if args.datadim == "3d" and args.upsample == "bilinear":
        args.upsample = "trilinear"
    
    if args.patch_shape is None:
        if args.datadim == '2d':
            args.patch_shape = [-1, -1]
        else:
            args.patch_shape = [-1, -1, -1]
            
    return args
