from .convgru import *
from .partial_unet import *
from .unet import *
from .attention import *
from .mulresunet import *
from .skip import *


def get_net(args, outchannel=1):
    if args.datadim in ['2d', '2.5d']:
        if args.net == 'unet':
            net = UNet(
                num_input_channels=args.inputdepth,
                num_output_channels=outchannel,
                filters=args.filters,
                upsample_mode=args.upsample,
                need_bias=True,
                act_fun=args.activation,
                last_act_fun=args.last_activation,
                dropout=args.dropout,
            )
        elif args.net == 'attmultiunet':
            net = AttMulResUnet2D(
                num_input_channels=args.inputdepth,
                num_output_channels=outchannel,
                num_channels_down=args.filters,
                upsample_mode=args.upsample,
                need_bias=True,
                act_fun=args.activation,
                last_act_fun=args.last_activation,
                dropout=args.dropout,
            )
        elif args.net == 'part':
            net = PartialUNet(args.inputdepth,
                              outchannel,
                              use_bn=True,
                              need_bias=True,
                              act_fun=args.activation,
                              dropout=args.dropout)
        else:
            net = MulResUnet(
                num_input_channels=args.inputdepth,
                num_output_channels=outchannel,
                num_channels_down=args.filters,
                num_channels_up=args.filters,
                num_channels_skip=args.skip,
                upsample_mode=args.upsample,
                need_bias=True,
                act_fun=args.activation,
                last_act_fun=args.last_activation,
                dropout=args.dropout,
            )
    else:  # 3D architecture
        if args.net == 'part':
            net = PartialUNet3D(args.inputdepth,
                                outchannel,
                                use_bn=True,
                                need_bias=True,
                                act_fun=args.activation,
                                dropout=args.dropout)
        elif args.net == 'skip':
            net = Skip3D(num_input_channels=args.inputdepth,
                         num_output_channels=outchannel,
                         num_channels_down=args.filters,
                         num_channels_up=args.filters,
                         num_channels_skip=args.skip,
                         upsample_mode=args.upsample,
                         need_bias=True,
                         act_fun=args.activation,
                         last_act_fun=args.last_activation,
                         dropout=args.dropout, )
        else:
            net = MulResUnet3D(
                num_input_channels=args.inputdepth,
                num_output_channels=outchannel,
                num_channels_down=args.filters,
                num_channels_up=args.filters,
                num_channels_skip=args.skip,
                upsample_mode=args.upsample,
                need_bias=True,
                act_fun=args.activation,
                last_act_fun=args.last_activation,
                dropout=args.dropout,
            )
    return net
