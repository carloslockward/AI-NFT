import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as Transforms
import numpy as np
from ai import AddGaussianNoise
from torch.nn.modules.loss import MSELoss, SmoothL1Loss, HuberLoss
from torch.autograd import Variable
from torch import autograd
import math
from tools import *


class DeepQCNN(nn.Module):
    def __init__(self, image_dim, num_channels, fc_arch, lr):
        super().__init__()
        self.image_dim = image_dim
        self.num_channels = num_channels
        self.loss = SmoothL1Loss()
        self.conv_layers = self.create_conv_layers()
        self.net = self.create_network_layers(self.conv_layers, fc_arch)
        self.apply(self._init_weights)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # optim.RMSprop(self.parameters(), lr=lr, alpha=0.95, eps=0.01)

        # self._init_weights(self.fc_layers)
        # self._init_weights(self.conv_layers)

    def forward(self, x):
        return self.net(x)

    def cnn_out_dim(self, cnn):
        with T.no_grad():
            return cnn(T.zeros(1, self.num_channels, *(self.image_dim))).shape[1]

    def _init_weights(self, m):

        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            # nn.init.xavier_normal_(m.weight)
            # m.bias.data.fill_(0.0001)
            # nn.init.xavier_normal_(m.weight.data)
            # nn.init.zeros_(m.bias)

    def create_network_layers(self, cnn, fc_layers):
        layers = [
            cnn,
            nn.Linear(
                self.cnn_out_dim(cnn),
                fc_layers[0],
                bias=True,
            ),
            nn.ReLU(),
        ]

        for i in range(len(fc_layers) - 1):
            l = fc_layers[i]
            l_1 = fc_layers[i + 1]
            if i + 2 == len(fc_layers):
                layers += [nn.Linear(l, l_1)]
            else:
                layers += [nn.Linear(l, l_1), nn.ReLU()]

        return nn.Sequential(*layers)

    def create_conv_layers(self):

        cnn = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        return cnn


class MyGAN:
    def __init__(
        self,
        latent_size,
        image_size,
        gen_image_size,
        batch_size,
        max_target=1.0,
        clip_d=0.01,
        clip_g=0.01,
    ) -> None:
        self.latent_size = latent_size
        self.image_size = image_size
        self.gen_image_size = gen_image_size
        self.batch_size = batch_size
        self.discriminator = self.create_discriminator()
        self.discriminator.apply(self.weights_init)
        self.generator = self.create_generator()
        self.generator.apply(self.weights_init)
        self.device = "cpu"
        self.max_target = max_target
        self.clip_d = clip_d
        self.clip_g = clip_g
        if self.image_size != self.gen_image_size:
            self.transform = Transforms.Compose(
                [Transforms.Resize(image_size), AddGaussianNoise(0, 0.2)]
            )
        else:
            self.transform = AddGaussianNoise(0, 0.2)
        if image_size > 1024:
            raise Exception(
                f"This architecture does not support images larger that 1024x1024. Image size: {image_size}"
            )

    def __null_transform(self, x):
        return x

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def to_device(self, device):
        self.device = device
        self.discriminator = self.discriminator.to(device, non_blocking=True)
        self.generator = self.generator.to(device, non_blocking=True)

    def create_generator(self):
        power = round(math.log(self.gen_image_size) / math.log(2))
        layer_size = 1024
        layers = [
            nn.ConvTranspose2d(
                self.latent_size,
                layer_size,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(layer_size),
            nn.ReLU(True),
        ]
        for _ in range(power - 3):
            layers.append(
                nn.ConvTranspose2d(
                    layer_size,
                    layer_size // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(layer_size // 2))
            layers.append(nn.ReLU(True))
            layer_size = layer_size // 2

        layers.append(
            nn.ConvTranspose2d(
                layer_size, 3, kernel_size=4, stride=2, padding=1, bias=False
            )
        )
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def create_discriminator(self):
        max_conv_layer = 1024
        power = round(math.log(self.image_size) / math.log(2))
        layers = [
            nn.Conv2d(
                3,
                max_conv_layer // 2 ** (power - 3),
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(max_conv_layer // 2 ** (power - 3)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for p in range(power - 4):
            layers.append(
                nn.Conv2d(
                    max_conv_layer // 2 ** (power - 3 - p),
                    max_conv_layer // 2 ** (power - 4 - p),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(max_conv_layer // 2 ** (power - 4 - p)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(
            nn.Conv2d(
                max_conv_layer // 2,
                max_conv_layer,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(max_conv_layer))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(
            nn.Conv2d(max_conv_layer, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )
        layers.append(nn.Flatten())
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def loss_func(self, pred, target):
        # return pred.mean()
        return F.binary_cross_entropy(pred, target)
        # criterion = nn.BCELoss()
        # return criterion(pred, target)

    def train_discriminator(self, real_images, opt_d):

        opt_d.zero_grad()

        # Pass real images through discriminator
        real_preds = self.discriminator(real_images)
        real_targets = T.ones(real_images.size(0), 1, device=self.device)
        if self.max_target != 1.0:
            real_targets = real_targets.clamp(None, self.max_target)

        real_loss = self.loss_func(real_preds, real_targets)
        real_score = T.mean(real_preds).item()
        # real_loss.backward()

        # Generate fake images
        latent = T.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.transform(self.generator(latent))

        # Pass fake images through discriminator
        fake_targets = T.zeros(fake_images.size(0), 1, device=self.device)
        fake_preds = self.discriminator(fake_images)
        fake_loss = self.loss_func(fake_preds, fake_targets)
        fake_score = T.mean(fake_preds).item()

        # fake_loss.backward()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()

        # d_loss = fake_loss - real_loss

        opt_d.step()

        # Weight clipping
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.clip_d, self.clip_d)

        return loss.item(), real_score, fake_score

    def train_generator(self, opt_g):
        # Clear generator gradients
        opt_g.zero_grad()

        # Generate fake images
        latent = T.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.transform(self.generator(latent))

        # Try to fool the discriminator
        preds = self.discriminator(fake_images)
        targets = T.ones(self.batch_size, 1, device=self.device)
        loss = self.loss_func(preds, targets)

        # Update generator weights
        loss.backward()
        opt_g.step()

        return loss.item()
