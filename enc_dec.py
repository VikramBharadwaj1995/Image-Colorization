import torch
import torch.nn as nn
import torch.nn.functional as F


class AE_conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(
                3, 3), padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=3),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(
                2, 2), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(
                1, 1), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(
                2, 2), padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(
                1, 1), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(
                2, 2), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(
                1, 1), padding=1),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            # note that here, we have the same number of output channels
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(
                1, 1), padding=1),
            nn.ReLU(True),
            nn.Upsample(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(
                1, 1), padding=1),
            nn.ReLU(True),
            nn.Upsample(),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(
                1, 1), padding=1),
            nn.ReLU(True),
            nn.Upsample(),
            nn.Conv2d(16, 2, kernel_size=(3, 3), stride=(
                3, 3), padding=1),
            nn.Tanh(),
            nn.Upsample()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
