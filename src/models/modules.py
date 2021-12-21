import torch
import torch.nn as nn
import torch.nn.functional as F


class Down_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_Block, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.conv_1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class Up_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_Block, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x_1, x_2):
        x_1 = self.conv_transpose(x_1)
        x = torch.cat([x_1, x_2], dim=1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvTranspose, self).__init__()
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_transpose(x)
