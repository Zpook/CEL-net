import torch
import torch.nn as nn
import torch.nn.functional as functional

from networks.blocks import AdaDoubleConv2d, AdaptiveFM
from . import BaseAdanet


class CELNet(BaseAdanet):
    def __init__(self, adaptive: bool = False):
        super().__init__()
        self.adaptive = adaptive

        self.adaConv1 = AdaDoubleConv2d(4, 32, adaptive)
        self.adaConv2 = AdaDoubleConv2d(32, 64, adaptive)
        self.adaConv3 = AdaDoubleConv2d(64, 128, adaptive)
        self.adaConv4 = AdaDoubleConv2d(128, 256, adaptive)
        self.adaConv5 = AdaDoubleConv2d(256, 512, adaptive)

        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.adaConv6 = AdaDoubleConv2d(512, 256, adaptive)

        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.adaConv7 = AdaDoubleConv2d(256, 128, adaptive)

        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.adaConv8 = AdaDoubleConv2d(128, 64, adaptive)

        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.adaConv9 = AdaDoubleConv2d(64, 32, adaptive)

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=12, kernel_size=1)
        self.ada10 = AdaptiveFM(12, 4)

    def forward(self, x):


        conv1 = self.adaConv1(x)
        pool1 = functional.max_pool2d(conv1, kernel_size=2)

        conv2 = self.adaConv2(pool1)
        pool1.detach()
        pool2 = functional.max_pool2d(conv2, kernel_size=2)

        conv3 = self.adaConv3(pool2)
        pool2.detach()
        pool3 = functional.max_pool2d(conv3, kernel_size=2)

        conv4 = self.adaConv4(pool3)
        pool3.detach()
        pool4 = functional.max_pool2d(conv4, kernel_size=2)

        conv5 = self.adaConv5(pool4)
        pool4.detach()

        up6 = self.up6(conv5)
        conv5.detach()
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.adaConv6(up6)
        up6.detach()

        up7 = self.up7(conv6)
        conv6.detach()
        up7 = torch.cat([up7, conv3], 1)
        conv3.detach()
        conv7 = self.adaConv7(up7)
        up7.detach()

        up8 = self.up8(conv7)
        conv7.detach()
        up8 = torch.cat([up8, conv2], 1)
        conv2.detach()
        conv8 = self.adaConv8(up8)
        up8.detach()

        up9 = self.up9(conv8)
        conv8.detach()
        up9 = torch.cat([up9, conv1], 1)
        conv1.detach()
        conv9 = self.adaConv9(up9)
        up9.detach()

        conv10 = self.conv10(conv9)
        if self.adaptive:
            conv10 = self.ada10(conv10)
        out = functional.pixel_shuffle(conv10, 2)

        return out

