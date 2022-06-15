"""
  FileName     [ acgan.py ]
  PackageName  [ DLCV Spring 2019 - ACGAN ]
  Synopsis     [ ACGAN Model, train method and generate method. ]

  Dataset: CelebA
  Input.shape: 64 * 64 * 3
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from matplotlib import pyplot as plt

import dataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-5, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--tag", type=str, help="name of the model")
parser.add_argument("--sample_interval", type=int, default=625, help="interval between image sampling")
parser.add_argument("--log_interval", type=int, default=10, help="interval between everytime logging the G/D loss.")
parser.add_argument("--save_interval", type=int, default=625, help="interval to save the models")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
print("CUDA: {}".format(cuda))
device = utils.selectDevice()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        # --------------------------------------------- #
        #  out.h = (in.h - 1) * stride + out_padding    #
        #       - 2 * in_padding + kernel_size          #
        #                                               #
        #  Notes:                                       #
        #    kernel_size = 5                            #
        #    out_padding = 3                            #
        #    in_padding  = 1                            #
        # --------------------------------------------- #
        super(Generator, self).__init__()

        self.linear = nn.Linear(102, 512 * 4 * 4)
        self.bn0    = nn.BatchNorm2d(512)
        self.relu0  = nn.ReLU(inplace=True)

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, opt.channels, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(64, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        gen = torch.cat((z, labels), dim=1)
        gen = self.linear(gen).view(-1, 512, 4, 4)
        gen = self.bn0(gen)
        gen = self.relu0(gen)

        img = self.conv_blocks(gen)

        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(opt.channels, 64, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output layers
        self.adversarial_layer = nn.Sequential(nn.Linear(512 * 4 * 4, 1), nn.Sigmoid())
        self.auxiliary_layer   = nn.Sequential(nn.Linear(512 * 4 * 4, 2), nn.Softmax())

    def forward(self, img):
        # print("Discriminator_in.shape: \t{}".format(img.shape))
        out = self.conv_blocks(img)
        # print("Discriminator_out.shape: \t{}".format(out.shape))
        out = out.view(out.shape[0], -1)
        # print("Discriminator_out.shape: \t{}".format(out.shape))
        
        validity = self.adversarial_layer(out)
        label = self.auxiliary_layer(out)

        return validity, label


# Loss functions
adversarial_loss = torch.nn.MSELoss().to(device)
auxiliary_loss = torch.nn.BCELoss().to(device)

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
feature = utils.faceFeatures[7]
dataset = dataset.CelebA("./hw3_data/face/", feature, transform=transforms.Compose([transforms.ToTensor()]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(number):
    """ Saves a grid of generated faces with [feature] and [not_feature] """
    # Sample noise
    z = FloatTensor(np.random.normal(0, 1, size=(10, opt.latent_dim)))
    z = torch.cat((z, z), dim=0)
    # print("z.shape: \t{}".format(z.shape))

    # Get labels ranging from 0 to n_classes for n rows
    labels = FloatTensor(np.array([[1 - num, num] for num in range(2) for _ in range(10)]))
    # print("labels.shape: \t{}".format(labels.shape))
    # print("labels: {}".format(labels))
    gen_imgs = generator(z, labels)
    
    save_image(gen_imgs.data, "./output/acgan/{}-{}/{}.png".format(opt.tag, feature, number), nrow=10, normalize=True)

def train(epoch):
    generator_loss = []
    discriminator_loss = []
    discriminator_acc  = []

    for i, (imgs, labels) in enumerate(dataloader, 1):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = FloatTensor(batch_size, 1).fill_(1.0)
        fake = FloatTensor(batch_size, 1).fill_(0.0)

        # Configure input
        real_imgs = imgs.type(FloatTensor)
        labels = labels.type(FloatTensor)
        # print("Label.shape: \t{}".format(labels.shape))

        # --------------------- #
        #  Train Generator      #
        # --------------------- #
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = FloatTensor(np.random.normal(0, 1, size=(batch_size, opt.latent_dim)))
        gen_labels = FloatTensor(np.random.randint(0, 2, size=(batch_size, 1)))
        gen_labels = torch.cat((1 - gen_labels, gen_labels), dim=1)
        
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)

        # Comment:
        # Adversarial loss: 
        #   Measure the loss where caused by generated images
        #   (because the G(z, class) seems too fake) 
        #   Need to minimize the loss of G(z, class)
        # Auxiliary loss:
        #   Measure P(class | G(z, class))
        #   Want to maximize Prob(), that is, minimized the difference between D(G(z, class)) and class

        g_loss.backward()
        optimizer_G.step()

        generator_loss.append(g_loss.item())

        # --------------------- #
        #  Train Discriminator  #
        # --------------------- #
        optimizer_D.zero_grad()

        # Loss for real images
        # real_pred, real_aux = discriminator(real_imgs)
        # d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels))

        # Loss for fake images
        # fake_pred, fake_aux = discriminator(gen_imgs.detach())
        # d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels))

        # Loss for Adversarial loss
        real_adv, real_aux = discriminator(real_imgs)
        fake_adv, fake_aux = discriminator(gen_imgs.detach())

        d_loss_adv = adversarial_loss(real_adv, valid) + adversarial_loss(fake_adv, fake)
        d_loss_aux = auxiliary_loss(real_aux, labels)

        # Total discriminator loss
        # d_loss = 0.5 * (d_real_loss + d_fake_loss)
        d_loss = d_loss_adv + d_loss_aux

        # Comment:
        # Adversarial loss: 
        #   Measure the loss where caused by difference sources
        #   Classify G(z, class) as fake and X as real.
        # Auxiliary loss:
        #   Measure P(class | X)
        #   Want to maximize Prob(), that is, minimized the difference between D(X) and class

        # Calculate discriminator class accuracy
        # pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        # gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        # d_acc = np.mean(np.argmax(pred, axis=1) == gt)
        cls_pred = real_aux.data.cpu().numpy()
        gt = labels.data.cpu().numpy()
        d_acc_real = np.mean(np.argmax(cls_pred, axis=1) == np.argmax(gt, axis=1))

        cls_pred = fake_aux.data.cpu().numpy()
        gt = gen_labels.data.cpu().numpy()
        d_acc_fake = np.mean(np.argmax(cls_pred, axis=1) == np.argmax(gt, axis=1))

        d_loss.backward()
        optimizer_D.step()

        discriminator_loss.append(d_loss.item())
        discriminator_acc.append((d_acc_real + d_acc_fake) / 2)

        batches_done = (epoch - 1) * len(dataloader) + i
        
        # 1. Logging
        if batches_done % opt.log_interval == 0:
            print("[Epoch %d] [Batch %d/%d] [D loss: %f, accR: %d%%, accF: %d%%] [G loss: %f]" % (
                    epoch, i, len(dataloader), d_loss.item(), 100 * d_acc_real, 100 * d_acc_fake, g_loss.item()))
        
        # 2. Sampling
        if batches_done % opt.sample_interval == 0:
            number = batches_done // opt.sample_interval
            sample_image(number)

        # 3. Saving
        if batches_done % opt.save_interval == 0:
            number   = batches_done // opt.save_interval

            savepath = "./models/acgan/{}-{}".format(opt.tag, feature)
            utils.saveModel(os.path.join(savepath, "generator_{}.pth".format(number)), generator)
            
            print("Model saved to: {}, iteration: {}".format(savepath, number))

    return generator_loss, discriminator_loss, discriminator_acc

def main():
    # Print Training Setting
    os.system('clear')
    print(opt)
    print("Choose feature: {}".format(feature))
    
    # Set the environment path
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./output/acgan", exist_ok=True)
    os.makedirs("./output/acgan/{}-{}".format(opt.tag, feature), exist_ok=True)

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/acgan", exist_ok=True)
    os.makedirs("./models/acgan/{}-{}".format(opt.tag, feature), exist_ok=True)
    
    g_loss_iteration = []
    d_loss_iteration = []
    d_acc_iteration  = []
    
    for epoch in range(1, opt.n_epochs):
        generator_loss, discriminator_loss, discriminator_acc = train(epoch)
        g_loss_iteration += generator_loss
        d_loss_iteration += discriminator_loss
        d_acc_iteration  += discriminator_acc

        if epoch > 5:
            threshold = -5 * len(dataloader)

            g_loss_iteration = g_loss_iteration[threshold: ]
            d_loss_iteration = d_loss_iteration[threshold: ]
            d_acc_iteration  = d_acc_iteration[threshold: ]
            
            x = np.arange(start=((epoch - 5) * len(dataloader) + 1), stop=(epoch * len(dataloader) + 1))
        else:
            x = np.arange(start=1, stop=(epoch * len(dataloader) + 1))

        # TODO: Create a better loss logging function
        plt.clf()
        plt.figure(figsize=(12.8, 7.2))
        plt.plot(x, g_loss_iteration, label='G loss', color='r', linewidth=0.25)
        plt.plot(x, d_loss_iteration, label='D loss', color='b', linewidth=0.25)
        plt.plot(x, d_acc_iteration, label='D Accuracy', color='y', linewidth=1)
        plt.legend(loc=0)
        plt.ylabel("Loss")
        plt.xlabel("Iteration(s)")
        plt.title("G / D Loss vs Iteration(s)")
        plt.savefig("ACGAN_Loss_Iteration.png")

if __name__ == "__main__":
    main()
