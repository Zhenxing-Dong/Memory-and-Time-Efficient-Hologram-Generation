import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        x = self.gamma * (x * Nx) + self.beta + x
        return x

class LFM(nn.Module):
    def __init__(self,dim,warm = False):
        super().__init__()
        self.to_feat = nn.Sequential(
            nn.Conv2d(dim, dim//2, 3, 1, 1),
            nn.LeakyReLU(0.1,inplace=True)
        )
        self.warm = None
        if warm == True:
            self.warm = nn.Sequential(nn.Conv2d(dim//2, dim//2, 3, 1, 1),
                                      nn.LeakyReLU(0.1,inplace=True),)
        self.mix = nn.Sequential(
            nn.Conv2d(dim//2, dim//2, 3, 1, 1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(dim//2, dim,3,1,1)
        )
        self.act = nn.Sigmoid()
        
    def forward(self,x):
        out = self.to_feat(x)
        if self.warm != None:
            out = self.warm(out)
        out = self.mix(out)
        out = self.act(out)
        out = out * x
        return out   
    
class CCMA(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(hidden_dim, dim, 3, 1, 1),
            # nn.LeakyReLU(0.1,inplace=True),
        )

    def forward(self, x):
        return self.ccm(x)   

class LAttBlock(nn.Module):
    def __init__(self, dim, ffn_dim,warm = False):
        super().__init__()
        self.lfm = LFM(dim,warm) 
        self.ccm = CCMA(dim, ffn_dim/dim)

    def forward(self, x):
        x = self.lfm(x)+x
        x = self.ccm(x)+x
        return x

class LMN(nn.Module):
    def __init__(self, dim, ffn_dim, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor
        
        self.norm = GRN(dim)

        self.to_feat = nn.Conv2d(upscaling_factor**2, dim, 3, 1, 1)

        self.feats1 = LAttBlock(dim, ffn_dim)
        self.feats2 = LAttBlock(dim, ffn_dim)
        self.feats3 = LAttBlock(dim, ffn_dim)
        self.feats4 = LAttBlock(dim, ffn_dim)

        self.to_img = nn.Conv2d(dim, upscaling_factor**2, 3, 1, 1)
        self.out = nn.Hardtanh(-math.pi, math.pi)
        self.up = nn.PixelShuffle(upscaling_factor)
        
    def forward(self,  x):
        h = self.to_feat(x)
        h = self.norm(h)
        h = self.feats1(h)
        h = self.feats2(h)
        h = self.feats3(h)
        h = self.feats4(h)
        h = self.to_img(h)+x
        h = self.up(h)
        return self.out(h)
    
class LMNwotanh(nn.Module):
    def __init__(self, dim, ffn_dim, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor
        
        self.norm = GRN(dim)

        self.to_feat = nn.Conv2d(upscaling_factor**2, dim, 3, 1, 1)

        self.feats1 = LAttBlock(dim, ffn_dim)
        self.feats2 = LAttBlock(dim, ffn_dim)
        self.feats3 = LAttBlock(dim, ffn_dim)
        self.feats4 = LAttBlock(dim, ffn_dim)

        self.to_img = nn.Conv2d(dim, upscaling_factor**2, 3, 1, 1)
        self.up = nn.PixelShuffle(upscaling_factor)
        
    def forward(self,  x):
        h = self.to_feat(x)
        h = self.norm(h)
        h = self.feats1(h)
        h = self.feats2(h)
        h = self.feats3(h)
        h = self.feats4(h)
        h = self.to_img(h)+x
        h = self.up(h)
        return h

class LMN4K(nn.Module):
    def __init__(self, dim, ffn_dim, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor
        
        self.norm = GRN(dim)

        self.to_feat = nn.Conv2d(upscaling_factor**2, dim, 3, 1, 1)

        self.feats1 = LAttBlock(dim, ffn_dim)
        self.feats2 = LAttBlock(dim, ffn_dim)

        self.to_img = nn.Conv2d(dim, upscaling_factor**2, 3, 1, 1)
        self.out = nn.Hardtanh(-math.pi, math.pi)
        self.up = nn.PixelShuffle(upscaling_factor)
        
    def forward(self,  x):
        h = self.to_feat(x)
        h = self.norm(h)
        h = self.feats1(h)
        h = self.feats2(h)
        h = self.to_img(h)+x
        h = self.up(h)
        return self.out(h)

class LMN4Kwotanh(nn.Module):
    def __init__(self, dim, ffn_dim, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor
        
        self.norm = GRN(dim)

        self.to_feat = nn.Conv2d(upscaling_factor**2, dim, 3, 1, 1)

        self.feats1 = LAttBlock(dim, ffn_dim)
        self.feats2 = LAttBlock(dim, ffn_dim)
        
        self.to_img = nn.Conv2d(dim, upscaling_factor**2, 3, 1, 1)
        self.up = nn.PixelShuffle(upscaling_factor)
        
    def forward(self,  x):
        h = self.to_feat(x)
        h = self.norm(h)
        h = self.feats1(h)
        h = self.feats2(h)
        h = self.to_img(h)+x
        h = self.up(h)
        return h