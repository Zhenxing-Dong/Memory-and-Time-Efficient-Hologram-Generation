"""
Neural holography:

This is the main script used for training the Holonet

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

Usage
-----

$ python train_holonet.py --channel=1 --run_id=experiment_1


"""
import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import configargparse
from tensorboardX import SummaryWriter

import utils.utils as utils
import utils.perceptualloss as perceptualloss

from propagation_model import ModelPropagate
from propagation_ASM import propagation_ASM
from holonet import *
from utils.augmented_image_loader import ImageLoader
from tqdm import tqdm
from CCNN.model import ccnncgh,ccnncgh_SR
from utils.utils import filter2d
import pynvml
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  #固定随机数种子
random.seed(seed)
# cudnn.benchmark = False
# cudnn.deterministic = True

# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--channel', type=int, default=1, help='red:0, green:1, blue:2, rgb:3')
p.add_argument('--run_id', type=str, default='', help='Experiment name', required=True)
p.add_argument('--proptype', type=str, default='ASM', help='Ideal propagation model')
p.add_argument('--generator_path', type=str, default='', help='Torch save of Holonet, start from pre-trained gen.')
p.add_argument('--save_path', type=str, default='checkpoints',help='Path to save trained models')
p.add_argument('--model_path', type=str, default='./models', help='Torch save CITL-calibrated model')
p.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
p.add_argument('--batch_size', type=int, default=1, help='Size of minibatch')
p.add_argument('--lr', type=float, default=1e-3, help='learning rate of Holonet weights')
p.add_argument('--scale_output', type=float, default=0.95,
               help='Scale of output applied to reconstructed intensity from SLM')
p.add_argument('--loss_fun', type=str, default='vgg-low', help='Options: mse, l1, si_mse, vgg, vgg-low')
p.add_argument('--purely_unet', type=utils.str2bool, default=False, help='Use U-Net for entire estimation if True')
p.add_argument('--model_lut', type=utils.str2bool, default=True, help='Activate the lut of model')
p.add_argument('--disable_loss_amp', type=utils.str2bool, default=True, help='Disable manual amplitude loss')
p.add_argument('--num_latent_codes', type=int, default=2, help='Number of latent codes of trained prop model.')
p.add_argument('--step_lr', type=utils.str2bool, default=False, help='Use of lr scheduler')
p.add_argument('--perfect_prop_model', type=utils.str2bool, default=False,
               help='Use ideal ASM as the loss function')
p.add_argument('--manual_aberr_corr', type=utils.str2bool, default=True,
               help='Divide source amplitude manually, (possibly apply inverse zernike of primal domain')
p.add_argument('--method',type=str,default="HOLONET",choices=["HOLONET","HOLONET_PS","HOLONET_SR","HOLONET_pyramid","complex","complex_SR"])
p.add_argument('--scale_factor',type=int,default=2,choices=[2,4])
p.add_argument('--res',type=int,default=0,choices=[0,1,2],help='0: (1088, 1920), 1: (2176, 3840), 2: (4352, 7680)')

# wait for GPU resources
# pynvml.nvmlInit()
# while(True):
#     handle = pynvml.nvmlDeviceGetHandleByIndex(3)
#     meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     if(meminfo.free/1024**2>11000):
#         break
#     print("reserved memory {} MiB, waiting for memory......".format(meminfo.free/1024**2))
#     time.sleep(300)    
# pynvml.nvmlShutdown()

# parse arguments
opt = p.parse_args()
channel = opt.channel
run_id = opt.run_id
loss_fun = opt.loss_fun
warm_start = opt.generator_path != ''
chan_str = ('red', 'green', 'blue')[channel]

# tensorboard setup and file naming
time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')
writer = SummaryWriter(f'run/{run_id}_{chan_str}_{time_str}')


##############
# Parameters #
##############

# units
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
df = 2**opt.res

# Propagation parameters
prop_dist = (7 * cm, 7 * cm, 7 * cm)[channel]
wavelength = (680 * nm, 520 * nm, 450 * nm)[channel]
feature_size = (3.74 * um, 3.74 * um)  # SLM pitch
homography_res = (int(880*df),int(1600*df))

# Training parameters
device = torch.device('cuda')
use_mse_init = False  # first 500 iters will be MSE regardless of loss_fun

# Image data for training
data_path = './DIV2K_train_HR'  #path for training data

if opt.model_path == '':
    opt.model_path = f'./models/{chan_str}.pth'

# resolutions need to be divisible by powers of 2 for unet
if opt.purely_unet:
    image_res = (1024, 2048)  # 10 down layers
else:
    image_res = (int(1088*df),int(1920*df))
    
###############
# Load models #
###############

# re-use parameters from CITL-calibrated model for out Holonet.
if opt.perfect_prop_model:
    final_phase_num_in = 2

    filter1 = None
    zernike_coeffs = None
    source_amplitude = None
    latent_codes = None
    u_t = None
    pad = True
    dtype = torch.float32

    Hbackward= propagation_ASM(torch.empty(1, 1, image_res[0], image_res[1]), feature_size=[feature_size[0], feature_size[1]],
                                wavelength=wavelength, z=prop_dist, linear_conv=pad,return_H=True,dtype = dtype).to(device)

# create new phase generator object
if opt.purely_unet:
    phase_generator = PhaseOnlyUnet(num_features_init=32).to(device)
else:
    if opt.method == "HOLONET":
        phase_generator = HoloNet(
            distance=prop_dist,
            wavelength=wavelength,
            feature_size=feature_size,
            res = image_res,
            zernike_coeffs=zernike_coeffs,
            source_amplitude=source_amplitude,
            initial_phase=InitialPhaseUnet(4, 16),
            final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=final_phase_num_in),
            manual_aberr_corr=opt.manual_aberr_corr,
            target_field=u_t,
            latent_codes=latent_codes,
            proptype=opt.proptype).to(device)
    elif opt.method == "HOLONET_PS":
        down = opt.scale_factor
        phase_generator = HoloNet_PixelShuffle(
        distance=prop_dist,
        wavelength=wavelength,
        feature_size=feature_size,
        res = image_res,
        zernike_coeffs=zernike_coeffs,
        source_amplitude=source_amplitude,
        initial_phase=FinalPhaseOnlyUnet(4,16,num_in=down**2,num_out=down**2),
        final_phase_only=FinalPhaseOnlyUnet_PS(4, 16, num_in=final_phase_num_in*(down**2),num_out=down**2),
        manual_aberr_corr=opt.manual_aberr_corr,
        target_field=u_t,
        latent_codes=latent_codes,
        proptype=opt.proptype,down = down).to(device)
    elif opt.method == "HOLONET_SR":
        down = opt.scale_factor
        phase_generator = HoloNet_SR(
        distance=prop_dist,
        wavelength=wavelength,
        feature_size=feature_size,
        res=image_res,
        zernike_coeffs=zernike_coeffs,
        source_amplitude=source_amplitude,
        initial_phase=FinalPhaseOnlyUnet(4,16,num_in=down**2,num_out=down**2),
        final_phase_only=FinalPhaseOnlyUnet_SR(3, 32, num_in=final_phase_num_in*(down**2),num_out=down**2),
        manual_aberr_corr=opt.manual_aberr_corr,
        target_field=u_t,
        latent_codes=latent_codes,
        proptype=opt.proptype,down_scale=down).to(device)
    elif opt.method == "HOLONET_pyramid":
        down = opt.scale_factor
        phase_generator = HoloNet_pyramid(
        distance=prop_dist,
        wavelength=wavelength,
        feature_size=feature_size,
        res = image_res,
        zernike_coeffs=zernike_coeffs,
        source_amplitude=source_amplitude,
        initial_phase=FinalPhaseOnlyUnet(4,16,num_in=down**2,num_out=down**2),
        final_phase_only=FinalPhaseOnlyUnet_SR(4, 16, num_in=final_phase_num_in*(down**2),num_out=down**2),
        manual_aberr_corr=opt.manual_aberr_corr,
        target_field=u_t,
        latent_codes=latent_codes,
        proptype=opt.proptype,down_scale=4).to(device)
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

if warm_start:
    phase_generator.load_state_dict(torch.load(opt.generator_path, map_location=device),strict=False)

phase_generator.train()  # generator to be trained


###################
# Set up training #
###################

# loss function
loss_fun = ['mse', 'l1', 'si_mse', 'vgg', 'ssim', 'vgg-low', 'l1-vgg'].index(loss_fun.lower())

if loss_fun == 0:        # MSE loss
    loss = nn.MSELoss()
elif loss_fun == 1:      # L1 loss
    loss = nn.L1Loss()
elif loss_fun == 2:      # scale invariant MSE loss
    loss = nn.MSELoss()
elif loss_fun == 3:      # vgg perceptual loss
    loss = perceptualloss.PerceptualLoss()
elif loss_fun == 5:
    loss = perceptualloss.PerceptualLoss(lambda_feat=0.025)
    loss_fun = 3

mse_loss = nn.MSELoss()

# upload to GPU
loss = loss.to(device)
mse_loss = mse_loss.to(device)

if loss_fun == 2:
    scaleLoss = torch.ones(1)
    scaleLoss = scaleLoss.to(device)
    scaleLoss.requires_grad = True

    optvars = [scaleLoss, *phase_generator.parameters()]
else:
    optvars = phase_generator.parameters()

# set aside the VGG loss for the first num_mse_epochs
if loss_fun == 3:
    vgg_loss = loss
    loss = mse_loss

# create optimizer
if warm_start:
    opt.lr /= 4
optimizer = optim.Adam(optvars, lr=opt.lr)

# loads images from disk, set to augment with flipping
image_loader = ImageLoader(data_path,
                           channel=channel,
                           batch_size=opt.batch_size,
                           image_res=image_res,
                           homography_res=homography_res,
                           shuffle=True,
                           vertical_flips=True,
                           horizontal_flips=True,
                           crop_to_homography=False)

num_mse_iters = 500
num_mse_epochs = 1 if use_mse_init else 0
opt.num_epochs += 1 if use_mse_init else 0

# learning rate scheduler
# if opt.step_lr:
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs-2, eta_min=1e-5)


#################
# Training loop #
#################
# torch.autograd.set_detect_anomaly(True)
best_loss = 1
for i in range(opt.num_epochs):
    # switch to actual loss function from mse when desired
    epoch_loss = []
    if i == num_mse_epochs:
        if loss_fun == 3:
            loss = vgg_loss

    for k, target in enumerate(image_loader):
        # cap iters on the mse epoch(s)
        if i < num_mse_epochs and k == num_mse_iters:
            break

        # get target image
        target_amp, target_res, target_filename = target
        target_amp = target_amp.to(device)

        # cropping to desired region for loss
        if homography_res is not None:
            target_res = homography_res
        else:
            target_res = target_res[0]  # use resolution of first image in batch

        optimizer.zero_grad()
        # forward model
        slm_amp, slm_phase = phase_generator(target_amp)
        predict_complex = torch.complex(torch.cos(slm_phase), torch.sin(slm_phase))
        if target_amp.shape[-2] > 1088:
            if filter1 is None:
                predict_complex, filter1 = filter2d(predict_complex,filter1,True)
            else:
                predict_complex = filter2d(predict_complex,filter1,False)
        output_complex = propagation_ASM(u_in=predict_complex, z=prop_dist, linear_conv=pad, feature_size=[feature_size[0], feature_size[1]],
                                    wavelength=wavelength,
                                    precomped_H=Hbackward,dtype = dtype)

        if loss_fun == 2:
            output_complex = output_complex * scaleLoss

        output_lin_intensity = torch.sum(output_complex.abs()**2 * opt.scale_output, dim=1, keepdim=True)

        output_amp = torch.pow(output_lin_intensity, 0.5)

        # VGG assumes RGB input, we just replicate
        if loss_fun == 3:
            target_amp = target_amp.repeat(1, 3, 1, 1)
            output_amp = output_amp.repeat(1, 3, 1, 1)

        # ignore the cropping and do full-image loss
        if 'nocrop' in run_id:
            loss_nocrop = loss(output_amp, target_amp)

        # crop outputs to the region we care about
        output_amp = utils.crop_image(output_amp, target_res, stacked_complex=False)
        target_amp = utils.crop_image(target_amp, target_res, stacked_complex=False)

        # force equal mean amplitude when checking loss
        if 'force_scale' in run_id:
            print('scale forced equal', end=' ')
            with torch.no_grad():
                scaled_out = output_amp * target_amp.mean() / output_amp.mean()
            output_amp = output_amp + (scaled_out - output_amp).detach()

        # loss and optimize
        loss_main = loss(output_amp, target_amp)
        if warm_start or opt.disable_loss_amp:
            loss_amp = 0
        else:
            # extra loss term to encourage uniform amplitude at SLM plane
            # helps some networks converge properly initially
            loss_amp = mse_loss(slm_amp.mean(), slm_amp)

        loss_val = ((loss_nocrop if 'nocrop' in run_id else loss_main)
                    + 0.1 * loss_amp)
        
        loss_val.backward()
        optimizer.step()

        epoch_loss.append(loss_val.item())

        # print loss
        ik = k + i * len(image_loader)
        if use_mse_init and i >= num_mse_epochs:
            ik += num_mse_iters - len(image_loader)
        print(f'iteration {ik}: {loss_val.item()}')
    
    #log    
    with torch.no_grad():
        writer.add_scalar('Loss', sum(epoch_loss)/len(epoch_loss),i+1)

    # save trained model
    if i >= 5:
        if not os.path.isdir(opt.save_path):
            os.mkdir(opt.save_path)
        torch.save(phase_generator.state_dict(),
                f'{opt.save_path}/{run_id}_{chan_str}_{time_str}_{i+1}.pth')
    
    #lr scheduler
    if opt.step_lr and (i+1)%5==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.5
writer.close() 
