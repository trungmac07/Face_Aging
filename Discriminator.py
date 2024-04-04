import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, img_size = 128, conv_dim = 64, c_dim = 5):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=5, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.02))

        for i in range(1, 7):
            layers.append(nn.Conv2d(conv_dim*i, conv_dim*2*i, kernel_size=5, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.02))


        kernel_size = int(img_size / np.power(2, 6))
        self.layers = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(conv_dim*2*6, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(conv_dim*2*6, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        res = self.layers(x)

        fake = self.conv1(x)
        label = self.conv2(res)

        return fake, label.view(label.size(0), label.size(1))