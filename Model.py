import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Generator import *
from Discriminator import *
from ReadImage import *

class StarGAN(nn.Module):
    def __init__(self, l_cls=1, l_rec=10, n_critic=1, device='cuda' if torch.cuda.is_available() else 'cpu', G=None, D=None):
        super().__init__()
        self.G = G if G != None else Generator()
        self.D = D if D != None else Discriminator()
        self.l_cls = l_cls
        self.l_rec = l_rec
        self.n_critic = n_critic
        self.device = device

    def discriminator_loss(self, real_res, predicted_res, real_labels, predicted_labels):
        real_res = real_res.float().to(self.device)
        predicted_res = predicted_res.float().to(self.device)
        real_labels = real_labels.float().to(self.device)
        predicted_labels = predicted_labels.float().to(self.device)
    
        return self.l_cls * F.cross_entropy(predicted_labels, real_labels) + F.binary_cross_entropy(predicted_res, real_res)

    def generator_loss(self, real_res, predicted_res, real_labels, predicted_labels, x, x_rec):
        real_res = real_res.float().to(self.device)
        predicted_res = predicted_res.float().to(self.device)
        real_labels = real_labels.float().to(self.device)
        predicted_labels = predicted_labels.float().to(self.device)
        x = x.float().to(self.device)
        x_rec = x_rec.float().to(self.device)
        
        return (self.l_cls * F.cross_entropy(predicted_labels, real_labels) +
                self.l_rec * torch.mean(torch.abs(x - x_rec)) -
                F.binary_cross_entropy(predicted_res, real_res))

    def fit(self, x_train, labels, n_epochs=10, batch_size=32):

        self.G.train()
        self.D.train()

        x_data_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size)
        labels_data_loader = torch.utils.data.DataLoader(labels, batch_size=batch_size)

        d_optimizer = torch.optim.Adam(self.D.parameters(),lr = 0.005)
        g_optimizer = torch.optim.Adam(self.G.parameters(),lr = 0.005)

        for epoch in range(n_epochs):

            i = 1
            for x_data, labels_data in zip(x_data_loader, labels_data_loader):
                
                print(f"Training on batch {i}")
                i+=1
                
                x = x_data.to(self.device)
                l = labels_data.to(self.device)
                

                # Discriminator training
                print(f"Training Discriminator Process:")
                random_labels = torch.randint(0, 5, (x.size(0),), device=self.device)
                random_l = F.one_hot(random_labels, num_classes=5).to(self.device)
                print(f"\tGenerating Fake Image")
                fake_img = self.G(x, random_l)

                d_train_data = torch.cat((fake_img, x), dim= 0)

                real_res = torch.cat((torch.zeros(x.size(0)), torch.ones(x.size(0)))).to(self.device)
                label_res = torch.cat([random_l, l], dim= 0)

                print(f"\tForwarding Discriminator")
                d_res, d_label = self.D(d_train_data)
                
                d_optimizer.zero_grad()
                d_loss = self.discriminator_loss(real_res, d_res, label_res, d_label)
                print(f"\tD_loss:{d_loss}")
                d_loss.backward()
                d_optimizer.step()

                # Generator training
                if epoch % self.n_critic == 0:
                    print("Training Generator")
                    
                    print("\tGenerating Fake Image")
                    x_fake = self.G(x, random_l)

                    print("\tForwarding Discriminator")
                    res, label = self.D(x_fake)

                    g_real_res = torch.zeros(x.size(0)).to(self.device)
                    x_fake2 = self.G(x_fake, l)

                    print("\tGenerating Fake Image")
                    x_fake2 = self.G(x_fake, l)
                    g_optimizer.zero_grad()
                    g_loss = self.generator_loss(g_real_res, res, random_l, label, x, x_fake2)
                    print(f"\tG_loss:{g_loss}")

                    g_loss.backward()
                    g_optimizer.step()

            print(f"Epoch [{epoch + 1}/{n_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

            # Save models
            torch.save(self.state_dict(), "./model/stargan.pth")
            torch.save(self.G.state_dict(), "./model/stargan_g.pth")
            torch.save(self.D.state_dict(), "./model/stargan_d.pth")





input_path = './10_images/'

x, label = GetDataBase(input_path)
label = torch.tensor(label)

model = StarGAN()
model.train()
model.fit(x_train=x, labels=label)

