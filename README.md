# Unsupervised 3D Seismic Data Reconstruction Based On Deep Prior

This repository contains codes and examples of the data interpolation schemes that leverages the deep prior paradigm.

#### Authors
[Fantong Kong](mailto:kft_upc@hotmail.com)<sup>1</sup>,
[Francesco Picetti](mailto:francesco.picetti@polimi.it)<sup>2</sup>,
Vincenzo Lipari<sup>2</sup>, Paolo Bestagini<sup>2</sup>,
Xiaoming Tang<sup>1</sup>, and Stefano Tubaro<sup>2</sup>

1: School of Geosciences - China University of Petroleum (East), Qingdao, China<br>
2: Dipartimento di Elettronica, Informazione e Bioingegneria - Politecnico di Milano, Italy

## Abstract
Irregularity and coarse spatial sampling of seismic data strongly affect the performances
of processing and imaging algorithms. Therefore, interpolation is a necessary pre-processing
step in most of the processing workflows. In this work, we propose a seismic data interpolation
method based on the deep prior paradigm: an ad-hoc Convolutional Neural Network is used as
a prior to solve the interpolation inverse problem, avoiding any costly and prone-to-overfitting training stage.
In particular, the proposed method leverages a multi resolution U-Net with 3D convolution
kernels exploiting correlations in 3D seismic data, at different scales in all directions.
Numerical examples on different corrupted synthetic and field datasets show the effectiveness
and promising features of the proposed approach.

The inverse problem is defined starting from the sampling equation:

![Sampling Equation](readme_img/sampling_equation.png)

that is solved using the deep prior paradigm by

![Deep Inverse Problem](readme_img/problem_prior.png)

The estimate of the true model is then obtained as the output of the optimized network:

![Output](readme_img/output.png)

The architecture we propose is the MultiResolution UNet:

![MultiRes UNet](./readme_img/multires.png)

## Setup and Code Organization
The code mainly relies on top of pytorch. You can recreate our conda environment named `dpi`
(acronym for "deep prior interpolation") through
```
conda create env -f environment.yml
``` 
Then, activate it with `source activate dpi` before running any example.<br>
**NOTE**: if you have initialized conda through `conda init`, use `conda activate dpi` instead.

This python project is organized as follows:
 - `main.py` is the main script that actually does the interpolation
 - `parameter.py` contains the run options that the main will parse as shell arguments. Check it out!
 - `architectures` contains pytorch implementations of the networks and loss functions
 - `data.py` contains data management utilities, such as data patch extraction.
 - `utils` contains some general purpose utilities

## Usage Examples
Here we report the example tests on the 3D hyperbolic data included in the paper.

     ```
     # Train from scratch with mask 1
     python main.py --imgname hyperbolic3d.npy --maskname hyperbolic3d_irregular_66_shot1.npy --datadim 3d --gain 40 --upsample nearest --epochs 3000
     # Train from scratch with mask 1 saving the optimized network weight
     python main.py --imgname hyperbolic3d.npy --maskname hyperbolic3d_irregular_66_shot1.npy --datadim 3d --gain 40 --upsample nearest --epochs 3000 --savemodel --outpath shot1
     # Train a network with mask 2 using as initial guess the optimization of mask 1
     python main.py --imgname hyperbolic3d.npy --maskname hyperbolic3d_irregular_66_shot2.npy --datadim 3d --gain 40 --upsample nearest --epochs 3000 --net load --netdir shot1/0_model.pth
     ```
    
#### Data preparation
We are glad you want to try our method on your data! To minimize the effort, keep in mind that:
 - The data dimensions are (*t,x,y*), and so are defined the patch shape and stride (during extraction).
 If you have 2D native datasets, please add an extra axis.
 - If you process the data in a 2.5D fashion, the tensors will be transposed in order to 
 tile the patches in the last dimension (as they are a channel).
 This procedure is automatic and should be reversed in the patch assembly in `data.reconstruct_patches`.
 - The subsampling mask can be made of 0 and 1; however we prefer to store the "decimated" version of the data, with NaN missing traces.
 This has the advantage of removing the ambiguity given by the zeros in the data and the zeros in the mask. 
 Nonetheless, our codes can take into account both ways.
 - We study the behaviour of the network as a nonlinear prior for the decimated data.
 Therefore we do not perform any preprocessing, a part from a scalar `--gain` for avoiding numerical errors.  

## Related Publications
 1. F. Kong, V. Lipari, F. Picetti, P. Bestagini, and S. Tubaro.
 "A Deep Prior Convolutional Autoencoder for Seismic Data Interpolation",
 in *European Association of Geophysicists and Engineers (EAGE) Annual Meeting*, 2020.
 [DOI](https://doi.org/10.3997/2214-4609.202011461)
 2. F. Kong, F. Picetti, V. Lipari, P. Bestagini, and S. Tubaro.
 "Deep prior-based seismic data interpolation via multi-res U-net",
 in *Society of Exploration Geophysicists (SEG) Annual Meeting*, 2020.
 [DOI](https://doi.org/10.1190/segam2020-3426173.1)
 3. F. Kong, F. Picetti, V. Lipari, P. Bestagini, X. Tang, and S. Tubaro.
 "Deep Prior Based Unsupervised Reconstruction of Irregularly Sampled Seismic Data",
 in *IEEE Geoscience and Remote Sensing Letters (GRSL)*, 2020.
 [DOI](https://doi.org/10.1109/LGRS.2020.3044455)
