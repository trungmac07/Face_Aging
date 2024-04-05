import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Generator import *
from Discriminator import *
from ReadImage import *

class StarGAN(nn.Module):
    def __init__(self, l_cls=1, l_rec=10, n_critic=1):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator()
        self.l_cls = l_cls
        self.l_rec = l_rec
        self.n_critic = n_critic

    def discriminator_loss(self, real_res, predicted_res, real_labels, predicted_labels):
        real_res = real_res.float()
        predicted_res = predicted_res.float()
        real_labels = real_labels.float()
        predicted_labels = predicted_labels.float()
        print(real_res.shape, predicted_res.shape, real_labels.shape, predicted_labels.shape)
        return self.l_cls * F.cross_entropy(predicted_labels, real_labels) - F.binary_cross_entropy(predicted_res, real_res)

    def generator_loss(self, real_res, predicted_res, real_labels, predicted_labels, x, x_rec):
        real_res = real_res.float()
        predicted_res = predicted_res.float()
        real_labels = real_labels.float()
        predicted_labels = predicted_labels.float()
        x = x.float()
        x_rec = x_rec.float()
        
        return (self.l_cls * F.cross_entropy(predicted_labels, real_labels) +
                self.l_rec * torch.mean(torch.abs(x - x_rec)) +
                F.binary_cross_entropy(predicted_res, real_res))

    def train(self, x_train, labels, n_epochs=10, batch_size=128, device='cuda' if torch.cuda.is_available() else 'cpu'):

        x_data_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size)
        labels_data_loader = torch.utils.data.DataLoader(labels, batch_size=batch_size)

        d_optimizer = torch.optim.Adam(self.D.parameters())
        g_optimizer = torch.optim.Adam(self.G.parameters())

        for epoch in range(n_epochs):
            for x_data, labels_data in zip(x_data_loader, labels_data_loader):
                x = x_data.to(device)
                x = x.permute(0, 3, 1, 2)
                l = labels_data.to(device)
                
                d_optimizer.zero_grad()

                # Discriminator training
                random_labels = torch.randint(0, 5, (batch_size,), device=device)
                random_l = F.one_hot(random_labels, num_classes=5)

                fake_img = self.G(x, random_l)

                d_train_data = torch.cat((fake_img, x), dim= 0)
                print(d_train_data.shape)

                real_res = torch.cat((torch.zeros(batch_size), torch.ones(batch_size))).to(device)
                label_res = torch.cat([random_l, l], dim= 0)

                d_res, d_label = self.D(d_train_data)

                d_loss = self.discriminator_loss(real_res, d_res, label_res, d_label)
                d_loss.backward()
                d_optimizer.step()

                # Generator training
                if epoch % self.n_critic == 0:
                    g_optimizer.zero_grad()
                    x_fake = self.G(x, random_l)
                    res, label = self.D(x_fake)

                    g_real_res = torch.zeros(batch_size).to(device)
                    x_fake2 = self.G(x_fake, l)

                    g_loss = self.generator_loss(g_real_res, res, random_l, label, x, x_fake2)
                    g_loss.backward()
                    g_optimizer.step()

            print(f"Epoch [{epoch + 1}/{n_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

            # Save models
            torch.save(self.state_dict(), "./model/startgan.pth")
            torch.save(self.G.state_dict(), "./model/startgan_g.pth")
            torch.save(self.D.state_dict(), "./model/startgan_d.pth")

input_path = './10_images/'

x, label = GetDataBase(input_path)
label = torch.tensor(label)

model = StarGAN()
model.train(x_train=x, labels=label)

