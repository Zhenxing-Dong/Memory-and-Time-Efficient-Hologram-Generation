import torch
from torch import nn, optim
import torch.fft
from CCNN.complexPyTorch.complexLayers import ComplexConvTranspose2d,ComplexConv2d
from CCNN.complexPyTorch.complexFunctions import complex_relu
from utils.pytorch_prototyping.pytorch_prototyping import  Unet
import math
from CCNN.propagation_ASM import propagation_ASM
from utils.utils import filter2d
from SR import *

class CDown(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1 = nn.Sequential(ComplexConv2d(in_channels, out_channels, 3, stride=2, padding=1))
    def forward(self, x):
        out1 = complex_relu((self.COV1(x)))
        return out1

class CDown2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1 = nn.Sequential(ComplexConv2d(in_channels, out_channels, 3, stride=2, padding=1))
    def forward(self, x):
        out1 = (self.COV1(x))
        return out1

class CUp(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1=nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))
    def forward(self, x):
        out1 = complex_relu((self.COV1(x)))
        return out1

class CUp2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1=nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))
    def forward(self, x):
        out1 =self.COV1(x)
        return out1

class pixelshuffle(nn.Module):
    def __init__(self,down):
        super().__init__()
        self.up = nn.PixelShuffle(down)
    
    def forward(self,x):
        out = self.up(x.real).type(torch.complex64) + \
                1j*self.up(x.imag).type(torch.complex64)
        return out
    
class pixelunshuffle(nn.Module):
    def __init__(self,up):
        super().__init__()
        self.down = nn.PixelUnshuffle(up)
        
    def forward(self,x):
        out = self.down(x.real).type(torch.complex64) + \
                1j*self.down(x.imag).type(torch.complex64)
        return out

class CCNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1 = CDown(1, 4)
        self.netdown2 = CDown(4, 8)
        self.netdown3 = CDown(8, 16)
        self.netdown4 = CDown(16, 32)

        self.netup4 = CUp(32, 16)
        self.netup3 = CUp(16, 8)
        self.netup2 = CUp(8, 4)
        self.netup1 = CUp2(4, 1)

    def forward(self, x):
        out1 = self.netdown1(x)
        out2 = self.netdown2(out1)
        out3 = self.netdown3(out2)
        out4 = self.netdown4(out3)

        out17 = self.netup4(out4)
        out18 = self.netup3(out17 + out3)
        out19 = self.netup2(out18 + out2)
        out20 = self.netup1(out19 + out1)

        predictphase = torch.atan2(out20.imag, out20.real)


        return predictphase
    
class CCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1 = CDown(1, 4)
        self.netdown2 = CDown(4, 8)
        self.netdown3 = CDown(8, 16)

        self.netup3 = CUp(16, 8)
        self.netup2 = CUp(8, 4)
        self.netup1 = CUp2(4, 1)

    def forward(self, x):
        out1 = self.netdown1(x)
        out2 = self.netdown2(out1)
        out3 = self.netdown3(out2)

        out18 = self.netup3(out3)
        out19 = self.netup2(out18 + out2)
        out20 = self.netup1(out19 + out1)

        holophase = torch.atan2(out20.imag, out20.real)
        return holophase

class CCNN3(nn.Module):
    def __init__(self,down=4):
        super().__init__()
        self.netdown1 = CDown(down**2, 32)
        self.netup1 = CUp2(32, down**2)

    def forward(self, x):
        out1 = self.netdown1(x)

        out20 = self.netup1(out1)+x

        predictphase = torch.atan2(out20.imag, out20.real)


        return predictphase

class CCNN4(nn.Module):
    def __init__(self,down=4):
        super().__init__()
        self.netdown1 = CDown(down**2,32)
        self.netup1 = CUp2(32, down**2)

    def forward(self, x):
        out1 = self.netdown1(x)
        
        out20 = self.netup1(out1) + x

        holophase = torch.atan2(out20.imag, out20.real)
        return holophase

class CCNN10(nn.Module):
    def __init__(self,down=4):
        super().__init__()
        self.netdown1 = CDown(down**2,32)
        self.netup1 = CUp2(32, down**2)

    def forward(self, x):
        out1 = self.netdown1(x)
        out20 = self.netup1(out1)

        holophase = torch.atan2(out20.imag, out20.real)
        return holophase

class ccnncgh(nn.Module):
    def __init__(self,z,pad,pitch,wavelength,res,H):
        super().__init__()
        self.ccnn1 = CCNN1()
        self.ccnn2 = CCNN2()
        self.z = -z
        self.pad = pad
        self.pitch = pitch
        self.wavelength = wavelength
        self.H = H
        if res[0] > 1088:
            _,self.filter = filter2d(torch.complex(torch.empty(1,1,*res),torch.empty(1,1,*res)).to(torch.device("cuda")),None,True)

        
    def forward(self, amp):
        target_complex = torch.complex(amp , torch.zeros_like(amp))
        predict_phase = self.ccnn1(target_complex)
        predict_complex = torch.complex(amp * torch.cos(predict_phase), amp * torch.sin(predict_phase))
        if amp.shape[-2] > 1088:
            predict_complex = filter2d(predict_complex,self.filter,False)
        slmfield = propagation_ASM(u_in=predict_complex, z=self.z, linear_conv=self.pad, feature_size=[self.pitch, self.pitch],
                                    wavelength=self.wavelength,
                                    precomped_H=self.H,dtype = self.H.dtype)

        holophase = self.ccnn2(slmfield)
        return predict_phase, holophase

class ccnncgh_SR(nn.Module):
    def __init__(self,z,pad,pitch,wavelength,res,H,down=4):
        super().__init__()
        self.ccnn1 = CCNN3(down)
        self.ccnn2 = CCNN4(down)
        if res[0] == 4352:
            self.ccnn3 = LMN4K(28,35,down)
        else:
            self.ccnn3 = LMN4K(24,30,down)
        self.up = nn.PixelShuffle(down)
        self.down = nn.PixelUnshuffle(down)
        self.z = -z
        self.pad = pad
        self.pitch = pitch
        self.wavelength = wavelength
        self.H = H
        if res[0] > 1088:
            _,self.filter = filter2d(torch.complex(torch.empty(1,1,*res),torch.empty(1,1,*res)).to(torch.device("cuda")),None,True)

        
    def forward(self, amp):
        target_complex = torch.complex(amp , torch.zeros_like(amp))
        target_complex = self.down(target_complex)
        predict_phase = self.ccnn1(target_complex)
        predict_phase = self.up(predict_phase)
        predict_complex = torch.complex(amp * torch.cos(predict_phase), amp * torch.sin(predict_phase))
        if amp.shape[-2] > 1088:
            predict_complex = filter2d(predict_complex,self.filter,False)
        slmfield = propagation_ASM(u_in=predict_complex, z=self.z, linear_conv=self.pad, feature_size=[self.pitch, self.pitch],
                                    wavelength=self.wavelength,
                                    precomped_H=self.H,dtype = self.H.dtype)
        slmfield = self.down(slmfield)
        holophase = self.ccnn2(slmfield)
        holophase = self.ccnn3(holophase)
        return predict_phase, holophase



class InitialPhaseUnet(nn.Module):
    """computes the initial input phase given a target amplitude"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d):
        super(InitialPhaseUnet, self).__init__()

        net = [Unet(1, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp):
        out_phase = self.net(amp)
        return out_phase


class FinalPhaseOnlyUnet(nn.Module):
    """computes the final SLM phase given a naive SLM amplitude and phase"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d, num_in=2):
        super(FinalPhaseOnlyUnet, self).__init__()

        net = [Unet(num_in, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp_phase):
        out_phase = self.net(amp_phase)
        return out_phase

class holonet(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_phase = InitialPhaseUnet(4, 16)
        self.final_phase_only = FinalPhaseOnlyUnet(4, 16)

    def forward(self, amp,phase,z,pad,pitch,wavelength,H):
        predict_phase = self.initial_phase(amp)
        predict_complex = torch.complex(amp * torch.cos(predict_phase), amp * torch.sin(predict_phase))
        slmfield = propagation_ASM(u_in=predict_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                                   wavelength=wavelength,
                                   precomped_H=H)
        slmamp=torch.pow(slmfield.real**2 + slmfield.imag**2, 0.5)
        slmphase=torch.atan2(slmfield.imag, slmfield.real)

        slm_amp_phase = torch.cat((slmamp, slmphase), -3)
        holophase = self.final_phase_only(slm_amp_phase)

        return holophase,predict_phase




class Down(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_factor):
        super().__init__()
        self.net1 = nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=dilation_factor,dilation=dilation_factor),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.net2 = nn.Sequential(
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, padding=0)
        )

    def forward(self, x):
        out1=self.net1(x)
        out2=self.skip(x)
        out3=out1+out2
        out4=self.net2(out3)
        out5=out4+out3
        return out5

class Up(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.net1 = nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1,output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.net2 = nn.Sequential(
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.skip = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, padding=0)
        )

    def forward(self, x):
        out1=self.net1(x)
        out2=self.skip(x)
        out3=out1+out2
        out4=self.net2(out3)
        out5=out4+out3
        return out5

class holoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1=Down(1,16,3)
        self.netdown2=Down(16,32,2)
        self.netdown3=Down(32,64,1)
        self.netdown4=Down(64,96,1)
        self.netup0=Up(96,64)
        self.netup1=Up(64,32)
        self.netup2=Up(32,16)
        self.netup3=Up(16,1)
        self.tan=torch.nn.Hardtanh(-math.pi, math.pi)

    def forward(self, amp,phase,z,pad,pitch,wavelength,H):
        out1=self.netdown1(amp)
        out2=self.netdown2(out1)
        out3=self.netdown3(out2)
        out4=self.netdown4(out3)

        out5=self.netup0(out4)
        out6 = self.netup1(out5+out3)
        out7 = self.netup2(out6+out2)
        out8 = self.netup3(out7+out1)
        out8=self.tan(out8)

        return out8,phase
