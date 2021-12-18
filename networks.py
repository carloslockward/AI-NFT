import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.modules.loss import MSELoss, SmoothL1Loss, HuberLoss



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