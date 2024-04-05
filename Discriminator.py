import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, img_size = 128, conv_dim = 64, c_dim = 5):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, 6):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(img_size / np.power(2, 6))
        self.layers = nn.Sequential(*layers)

        self.conv1 = nn.Sequential()
        self.conv1.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=2, padding=1, bias=False))
        self.conv1.append(nn.Sigmoid())

        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        x = x.float()  # Cast input tensor to float32

        res = self.layers(x)

        fake = self.conv1(res)
        label = self.conv2(res)

        return fake.view(fake.size(0)), label.view(label.size(0), label.size(1))