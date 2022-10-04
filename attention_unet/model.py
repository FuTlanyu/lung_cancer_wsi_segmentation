""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        self.up1 = Up(1024, 512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv1 = DoubleConv(1024,512)
        
        self.up2 = Up(512, 256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv2 = DoubleConv(512,256)
        
        self.up3 = Up(256, 128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = DoubleConv(256,128)
        
        self.up4 = Up(128, 64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv4 = DoubleConv(128,64)
        
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        d5 = self.up1(x5)
        e4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv1(d5)
        
        d4 = self.up2(d5)
        e3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv2(d4)
        
        d3 = self.up3(d4)
        e2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.up4(d3)
        e1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv4(d2)
        
        
        logits = self.outc(d2)
        return logits

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out
