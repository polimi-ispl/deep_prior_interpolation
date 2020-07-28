#!/bin/sh
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# 1.Process the 3D irregular hyperbolic data from scratch  
# python "$SCRIPTPATH/main.py" \
# --imgdir /nas/home/fkong/code/dippublic/deep_prior_interpolation/data/ \
# --outdir /nas/home/fkong/code/dippublic/deep_prior_interpolation/data/hyperbolic/ \
# --imgname hyperbolic3d.npy --maskname hyperbolic3d_irregular_66_shot2.npy \
# --datadim 3d --gpu 3 --gain 40 --filters 16 32 64 128 256 \
# --skip 16 32 64 128 --noise_dist u  \
# --inputdepth 64 --upsample nearest --inittype xavier --savemodel \
# --epochs 500 --lr 1e-3 --save_every 20 \

# 2.Process the 3D irregular hyperbolic data from last shot gather
# python "$SCRIPTPATH/main.py" \
# --imgdir /nas/home/fkong/code/dippublic/deep_prior_interpolation/data/ \
# --outdir /nas/home/fkong/code/dippublic/deep_prior_interpolation/data/hyperbolic/ \
# --imgname hyperbolic3d.npy --maskname hyperbolic3d_irregular_66_shot2.npy \
# --datadim 3d --gpu 3 --gain 40 --filters 16 32 64 128 256 \
# --skip 16 32 64 128 --noise_dist u  \
# --inputdepth 64 --upsample nearest --inittype xavier --savemodel \
# --epochs 500 --lr 1e-3 --save_every 20 \
# --net load --netdir /nas/home/fkong/code/deep_image_prior/data/weakdata/td67By/0_latest_net.pth \
# /nas/home/fkong/code/deep_image_prior/data/weakdata/td67By/1_latest_net.pth \
# /nas/home/fkong/code/deep_image_prior/data/weakdata/td67By/2_latest_net.pth

# 3.Process the 3D regular marmousi data using 2D common-unet model 
# python "$SCRIPTPATH/main.py" \
# --imgdir /nas/home/fkong/code/dippublic/deep_prior_interpolation/data/ \
# --outdir /nas/home/fkong/code/dippublic/deep_prior_interpolation/data/ \
# --imgname Marmousi.npy --maskname Marmousi_regular_050.npy \
# --datadim 2.5d --gpu 0 --gain 1e8 --model unet \
# --inputdepth 64 --upsample bilinear --inittype xavier \
# --epochs 2000 --lr 1e-3 --save_every 4000 \
# --slice YT --imgchannel 4  

# 4.Process the 3D regular marmousi data using 2D multi-unet model 
# python "$SCRIPTPATH/main.py" \
# --imgdir /nas/home/fkong/code/dippublic/deep_prior_interpolation/data/ \
# --outdir /nas/home/fkong/code/dippublic/deep_prior_interpolation/data/ \
# --imgname Marmousi.npy --maskname Marmousi_regular_050.npy \
# --datadim 2.5d --gpu 5 --gain 1e8 \
# --inputdepth 64 --upsample bilinear --inittype xavier \
# --epochs 2000 --lr 1e-3 --save_every 4000 \
# --slice YT --imgchannel 4  

# 5.Process the 3D regular marmousi data  
# python "$SCRIPTPATH/main.py" \
# --imgdir /nas/home/fkong/code/deep_image_prior/data/ \
# --outdir /nas/home/fkong/code/deep_image_prior/data/ \
# --imgname Marmousi.npy --maskname Marmousi_regular_050.npy \
# --datadim 3d --gpu 5 --gain 1e8 \
# --inputdepth 64 --upsample nearest --inittype xavier \
# --epochs 2000 --lr 1e-3 --save_every 4000 

# 6.Process the 3D neterlandsf3 migration image  
# python "$SCRIPTPATH/main.py" \
# --imgdir /nas/home/fkong/code/deep_image_prior/data/ \
# --outdir /nas/home/fkong/code/deep_image_prior/data/neterland/ \
# --imgname neterlandsf3.npy --maskname neterlandsf3_irregular_050_enzo.npy \
# --datadim 3d --gpu 1 --gain 0.003 --noise_dist n \
# --inputdepth 64 --upsample trilinear --inittype xavier \
# --epochs 3000 --lr 1e-3 --save_every 4000 

# 7.Process the 2D regular hyperbolic data(source-receiver)
# python "$SCRIPTPATH/main.py" \
# --imgdir /nas/home/fkong/data/seismic/interpolation/deep-prior/data/source/simple \
# --outdir /nas/home/fkong/data/seismic/interpolation/deep-prior/application/source/simple \
# --imgname hyperbolic3d_75.npy --maskname hyperbolic3d_75_mask.npy \
# --datadim 3d --gpu 5 --gain 25 --need_sigmoid --adirandel 0.3 \
# --inputdepth 64 --upsample trilinear --inittype xavier \
# --epochs 2000 --lr 1e-3 --save_every 4000 

# 8. Process the 2D sonic waveform
# python "$SCRIPTPATH/main.py" \
# --imgdir /nas/home/fkong/data/waveform/fast_qua/ \
# --outdir /nas/home/fkong/data/waveform/fast_qua/result/ \
# --imgname rec_0_ori.npy rec_1_ori.npy rec_2_ori.npy rec_3_ori.npy rec_4_ori.npy \
# rec_5_ori.npy rec_6_ori.npy rec_7_ori.npy rec_8_ori.npy \
# rec_9_ori.npy rec_10_ori.npy rec_11_ori.npy \
# --maskname rec_0_del6.npy rec_1_del6.npy rec_2_del6.npy rec_3_del6.npy rec_4_del6.npy \
# rec_5_del6.npy rec_6_del6.npy rec_7_del6.npy rec_8_del6.npy \
# rec_9_del6.npy rec_10_del6.npy rec_11_del6.npy \
# --datadim 2d --gpu 4 --gain 0.02 \
# --inputdepth 64 --upsample bilinear --inittype xavier \
# --epochs 2000 --lr 1e-3 --save_every 2000