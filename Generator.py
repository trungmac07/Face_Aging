import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, device):
        super(ResidualBlock, self).__init__()
        self.device = device
        self.layers = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False,device=self.device),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True,device=self.device),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False,device=self.device),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True,device=self.device),
        )

    def forward(self, x):
        return x + self.layers(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device

        self.layers = nn.Sequential()

        #3+c_dim -> 64
        self.layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False,device=self.device))
        self.layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True,device=self.device))
        self.layers.append(nn.ReLU(inplace=True))

        #encoder
        #64 -> 128
        self.layers.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False,device=self.device))
        self.layers.append(nn.InstanceNorm2d(conv_dim*2, affine=True, track_running_stats=True,device=self.device))
        self.layers.append(nn.ReLU(inplace=True))

        #128 -> 256
        self.layers.append(nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1, bias=False,device=self.device))
        self.layers.append(nn.InstanceNorm2d(conv_dim*4, affine=True, track_running_stats=True,device=self.device))
        self.layers.append(nn.ReLU(inplace=True))

        #transformation layers
        for i in range(6):
            self.layers.append(ResidualBlock(conv_dim*4,conv_dim*4,device=self.device))

        #decoder
        #256 -> 128
        self.layers.append(nn.ConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False,device=self.device))
        self.layers.append(nn.InstanceNorm2d(conv_dim*2, affine=True, track_running_stats=True,device=self.device))
        self.layers.append(nn.ReLU(inplace=True))

        #128 -> 64
        self.layers.append(nn.ConvTranspose2d(conv_dim*2, conv_dim, kernel_size=4, stride=2, padding=1, bias=False,device=self.device))
        self.layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True,device=self.device))
        self.layers.append(nn.ReLU(inplace=True))

        #64 -> 3
        self.layers.append(nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=False,device=self.device))
        self.layers.append(nn.Tanh())

    def forward(self, x, c):
        #x.size = 1x3x300x300 
        #c.size = 1x5x300x300
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))

        x = x.float()  # Cast input tensor to float32
        c = c.float()  # Cast c tensor to float32

        new_x = torch.cat([x, c], dim=1).to(self.device)

        return self.layers(new_x)   



     
        



