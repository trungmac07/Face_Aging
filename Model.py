import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable

from Generator import *
from Discriminator import *

class StarGAN(nn.Module):
    def __init__(self, l_cls = 1, l_rec = 10, n_critic = 1):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator()
        self.l_cls = l_cls
        self.l_rec = l_rec
        self.n_critic = n_critic
        

    def discriminator_loss(self, real_res, predicted_res, real_labels, predicted_labels):
        return self.l_cls * F.cross_entropy(real_labels, predicted_labels) - F.binary_cross_entropy(real_res, predicted_res)


    def generator_loss(self, real_res, predicted_res, real_labels, predicted_labels, x, x_rec):
        return self.l_cls * F.cross_entropy(real_labels, predicted_labels) + self.l_rec * torch.mean(torch.abs(x - x_rec)) + F.binary_cross_entropy(real_res, predicted_res)

    
    def train(self, x_train, labels, n_epochs = 10, batch_size = 128):
        
        x_data = torch.utils.data.DataLoader(x_train, batch_size = batch_size)
        labels_data = torch.utils.data.DataLoader(labels, batch_size = batch_size)
        d_optimizer = torch.optim.Adam(self.D.parameters())
        g_optimizer = torch.optim.Adam(self.G.parameters())

        for i in range(n_epochs):
            x = Variable(x_data)
            l = Variable(labels_data)
            d_optimizer.zero_grad()

            #Discriminator training
            random_l = [F.one_hot(torch.randint(0, 5, (1,)), num_classes=5).squeeze() for i in range(10)]
            random_l = torch.stack(random_l)

            fake_img = self.G(x, random_l)
            
            d_train_data = [fake_img,x]
            d_train_data = torch.stack(d_train_data)


            real_res = torch.Tensor(128 * [0] + 128 * [1])
            label_res = [random_l, l]
            label_res = torch.stack(label_res)

            d_res, d_label = self.D.forward(d_train_data)

            d_loss = self.discriminator_loss(real_res, d_res, label_res, d_label)
            d_loss.backward()
            d_optimizer.step()

            #Generator training
       
            if(i % self.n_critic == 0):
                g_optimizer.zero_grad()
                x_fake = self.G(x, random_l)
                res,label = self.D.forward(x_fake)

                g_real_res = torch.Tensor(128 * [0])
                x_fake2 = self.G(x_fake, l)

                g_loss = self.generator_loss(g_real_res, res, random_l, label, x, x_fake2)
                g_loss.backward()
                g_optimizer.step()

            torch.save(self.state_dict(), "./model/startgan.pth")
            torch.save(self.G.state_dict(), "./model/startgan_g.pth")
            torch.save(self.D.state_dict(), "./model/startgan_d.pth")



        

model = StarGAN()
model.train(1,2,3)

        