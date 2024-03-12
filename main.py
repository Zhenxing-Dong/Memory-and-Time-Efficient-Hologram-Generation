"""
Neural holography:

This is the main executive script used for the phase generation using Holonet/UNET or
                                                     optimization using (GS/DPAC/SGD) + camera-in-the-loop (CITL).

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

@article{Peng:2020:NeuralHolography,
author = {Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein},
title = {{Neural Holography with Camera-in-the-loop Training}},
journal = {ACM Trans. Graph. (SIGGRAPH Asia)},
year = {2020},
}

-----

$ python main.py --channel=0 --algorithm=HOLONET --root_path=./phases --generator_dir=./pretrained_models
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import configargparse
from torch.utils.tensorboard import SummaryWriter

import utils.utils as utils
from utils.augmented_image_loader import ImageLoader
from propagation_model import ModelPropagate
from utils.modules import SGD, GS, DPAC, PhysicalProp
from holonet import *
from CCNN.model import CCNN1,CCNN2,ccnncgh,ccnncgh_SR
from propagation_ASM import propagation_ASM
from thop import profile,clever_format
from ultralytics.utils.ops import Profile
import numpy as np
import time

# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--channel', type=int, default=1, help='Red:0, green:1, blue:2')
p.add_argument('--prop_model', type=str, default='ASM', help='Type of propagation model, ASM or model')
p.add_argument('--root_path', type=str, default='./out', help='Directory where optimized phases will be saved.')
p.add_argument('--data_path', type=str, default='./DIV2K_train_HR', help='Directory for the dataset')
p.add_argument('--generator_dir', type=str, default='./checkpoints',
               help='Directory for the pretrained holonet/unet network')
p.add_argument('--prop_model_dir', type=str, default='./calibrated_models',
               help='Directory for the CITL-calibrated wave propagation models')
p.add_argument('--citl', type=utils.str2bool, default=False, help='Use of Camera-in-the-loop optimization with SGD')
p.add_argument('--experiment', type=str, default='', help='Name of experiment')
p.add_argument('--lr', type=float, default=8e-3, help='Learning rate for phase variables (for SGD)')
p.add_argument('--lr_s', type=float, default=2e-3, help='Learning rate for learnable scale (for SGD)')
p.add_argument('--num_iters', type=int, default=500, help='Number of iterations (GS, SGD)')
p.add_argument('--res', type=int, default=1, help='Resolution of the SLM, 0:1088p, 1:4K, 2:8K')
p.add_argument('--method',type=str,default="HOLONET",choices=["GS","SGD","DPAC","UNET",
                                                              "HOLONET","HOLONET_PS","HOLONET_SR",
                                                              "HOLONET_pyramid","complex","complex_SR"])
p.add_argument('--scale_factor',type=int,default=2,choices=[2,4])
p.add_argument('--checkpoint', type= str,required = True)
# parse arguments
opt = p.parse_args()
run_id = f'{opt.experiment}_{opt.method}_{opt.prop_model}'  # {algorithm}_{prop_model} format
if opt.citl:
    run_id = f'{run_id}_citl'

channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]

print(f'   - optimizing phase with {opt.method}/{opt.prop_model} ... ')
if opt.citl:
    print(f'    - with camera-in-the-loop ...')

# Hyperparameters setting
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist = (7 * cm, 7 * cm, 7 * cm)[channel]  # propagation distance from SLM plane to target plane
wavelength = (680 * nm, 520 * nm, 450 * nm)[channel]  # wavelength of each color
feature_size = (3.74 * um, 3.74 * um)  # SLM pitch
if opt.res == 0:
    slm_res = (1088, 1920)  # resolution of SLM
    image_res = (1088, 1920)
    roi_res = (880, 1600)  # regions of interest (to penalize for SGD)
elif opt.res == 1:
    slm_res = (2176,3840)# resolution of SLM
    image_res = (2176,3840)#
    roi_res = (1760,3200)# 
elif opt.res == 2:
    slm_res = (4352,7680)# resolution of SLM
    image_res = (4352,7680)#
    roi_res = (3520,6400)# 
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
device = torch.device('cuda')  # The gpu you are using

# Options for the algorithm
loss = nn.MSELoss().to(device)  # loss functions to use (try other loss functions!)
s0 = 1.0  # initial scale
root_path = os.path.join(opt.root_path, run_id, chan_str)  # path for saving out optimized phases

# Tensorboard writer
summaries_dir = os.path.join(root_path, 'summaries')
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(summaries_dir)

# Hardware setup for CITL
if opt.citl:
    camera_prop = PhysicalProp(channel, laser_arduino=True, roi_res=(roi_res[1], roi_res[0]), slm_settle_time=0.12,
                               range_row=(220, 1000), range_col=(300, 1630),
                               patterns_path=f'F:/citl/calibration',
                               show_preview=True)
else:
    camera_prop = None

# Simulation model
if opt.prop_model == 'ASM':
    propagator = propagation_ASM  # Ideal model

elif opt.prop_model.upper() == 'MODEL':
    blur = utils.make_kernel_gaussian(0.85, 3)
    propagator = ModelPropagate(distance=prop_dist,  # Parameterized wave propagation model
                                feature_size=feature_size,
                                wavelength=wavelength,
                                blur=blur).to(device)

    # load CITL-calibrated model
    propagator.load_state_dict(torch.load(f'{opt.prop_model_dir}/{chan_str}.pth', map_location=device))
    propagator.eval()


# Select Phase generation method, algorithm
if opt.method == 'SGD':
    phase_only_algorithm = SGD(prop_dist, wavelength, feature_size, opt.num_iters, roi_res, root_path,
                               opt.prop_model, propagator, loss, opt.lr, opt.lr_s, s0, opt.citl, camera_prop, writer, device)
elif opt.method == 'GS':
    phase_only_algorithm = GS(prop_dist, wavelength, feature_size, opt.num_iters, root_path,
                              opt.prop_model, propagator, writer, device)
elif opt.method == 'DPAC':
    phase_only_algorithm = DPAC(prop_dist, wavelength, feature_size, opt.prop_model, propagator, device)
elif  'HOLONET' in opt.method or 'complex' in opt.method:
    if opt.method == "HOLONET":
        phase_only_algorithm = HoloNet(prop_dist, wavelength, feature_size, image_res,initial_phase=InitialPhaseUnet(4, 16),
                                   final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=2)).to(device)
    elif opt.method == "HOLONET_PS":
        down = opt.scale_factor
        phase_only_algorithm = HoloNet_PixelShuffle(
            distance=prop_dist,
            wavelength=wavelength,
            feature_size=feature_size,
            initial_phase=FinalPhaseOnlyUnet(4,16,num_in=down**2,num_out=down**2),
            final_phase_only=FinalPhaseOnlyUnet_PS(4, 16, num_in=2*(down**2),num_out=down**2),down = down).to(device)
    elif opt.method == "HOLONET_SR":
        down = opt.scale_factor
        phase_only_algorithm = HoloNet_SR(
            distance=prop_dist,
            wavelength=wavelength,
            feature_size=feature_size,
            res=image_res,
            initial_phase=FinalPhaseOnlyUnet(4,16,num_in=down**2,num_out=down**2),
            final_phase_only=FinalPhaseOnlyUnet_SR(4, 16, num_in=2*(down**2),num_out=down**2),down_scale=down).to(device)
    elif opt.method == "HOLONET_pyramid":
        down = opt.scale_factor
        phase_only_algorithm = HoloNet_pyramid(
            distance=prop_dist,
            wavelength=wavelength,
            feature_size=feature_size,
            res=image_res,
            initial_phase=FinalPhaseOnlyUnet(4,16,num_in=down**2,num_out=down**2),
            final_phase_only=FinalPhaseOnlyUnet_SR(4, 16, num_in=2*(down**2),num_out=down**2),down_scale=4).to(device)
    elif opt.method == "complex":
        pad = True
        dtype = torch.float32
        Hforward= propagation_ASM(torch.empty(1, 1, int(image_res[0]), int(image_res[1])), feature_size=[feature_size[0], feature_size[1]],
                                wavelength=wavelength, z=-prop_dist, linear_conv=pad,return_H=True,dtype = dtype).to(device)
        phase_generator = ccnncgh(z=prop_dist, wavelength=wavelength, pitch=[feature_size[0], feature_size[1]],res=image_res,pad=pad,H = Hforward).to(device)
    elif opt.method == "complex_SR":
        pad = True
        dtype = torch.float32
        down = opt.scale_factor
        Hforward= propagation_ASM(torch.empty(1, 1, int(image_res[0]), int(image_res[1])), feature_size=[feature_size[0], feature_size[1]],
                                wavelength=wavelength, z=-prop_dist, linear_conv=pad,return_H=True,dtype = dtype).to(device)
        phase_generator = ccnncgh_SR(z=prop_dist, wavelength=wavelength, pitch=[feature_size[0], feature_size[1]],res=image_res,pad=pad,H = Hforward,down = down).to(device)

    model_path = os.path.join(opt.generator_dir, f'{opt.checkpoint}')
elif opt.method == 'UNET':
    phase_only_algorithm = PhaseOnlyUnet(num_features_init=32).to(device)
    model_path = os.path.join(opt.generator_dir, f'unet20_{chan_str}.pth')
    image_res = (1024, 2048)


if 'NET' in opt.method or 'complex' in opt.method:
    checkpoint = torch.load(model_path)
    phase_only_algorithm.load_state_dict(checkpoint)
    phase_only_algorithm.eval()
    phase_only_algorithm.cuda()



# Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)
image_loader = ImageLoader(opt.data_path, channel=channel,
                           image_res=image_res, homography_res=roi_res,
                           crop_to_homography=False,
                           shuffle=False, vertical_flips=False, horizontal_flips=False)

random_input = torch.rand(1,1,*slm_res).to(device)
for _ in range(50):
    with torch.no_grad():
        _,_ = phase_only_algorithm(random_input)
times = []
# Loop over the dataset
for k, target in enumerate(image_loader):
    # get target image
    target_amp, target_res, target_filename = target
    target_path, target_filename = os.path.split(target_filename[0])
    target_idx = target_filename.split('_')[-1]
    target_amp = target_amp.to(device)
    print(target_amp.shape)
    print(target_idx)

    # if you want to separate folders by target_idx or whatever, you can do so here.
    phase_only_algorithm.init_scale = s0 * utils.crop_image(target_amp, roi_res, stacked_complex=False).mean()
    phase_only_algorithm.phase_path = os.path.join(root_path)

    # run algorithm (See algorithm_modules.py and algorithms.py)
    if opt.method in ["DPAC", "HOLONET", "UNET","HOLONET_PS","HOLONET_SR",
                                "HOLONET_pyramid","complex","complex_SR"]:
        # direct methods
        with torch.no_grad():
            #compute inference time
            with Profile() as dt:
                dummy_output, final_phase = phase_only_algorithm(target_amp)
            times.append(dt.t)
    else:
        # iterative methods, initial phase: random guess
        init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to(device)
        final_phase = phase_only_algorithm(target_amp, init_phase)

    print(final_phase.shape)

    # save the final result somewhere.
    phase_out_8bit = utils.phasemap_8bit(final_phase.cpu().detach(), inverted=True)

    utils.cond_mkdir(root_path)
    cv2.imwrite(os.path.join(root_path, f'{target_idx}.png'), phase_out_8bit)
print('='*20)
print("average time (s):",sum(times)/len(times))
dummy_input = torch.rand(1,1,*image_res).cuda()
macs, params = profile(phase_only_algorithm, inputs=(dummy_input,), verbose=0)
macs, params = clever_format([macs, params], "%.3f")
print("Flops:", macs)
print("Parameters:", params)
print(f'    - Done, result: --root_path={root_path}')
