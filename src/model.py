import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, input_length, output_length, filter_number=16):
        super(UNet, self).__init__()

        self.conv_1_1 = Conv(input_length, filter_number)
        self.conv_1_2 = Conv(filter_number, filter_number)
        self.down_1 = Down_Block(filter_number, filter_number*2)
        self.down_2 = Down_Block(filter_number*2, filter_number*4)
        self.down_3 = Down_Block(filter_number*4, filter_number*8)
        self.down_4 = Down_Block(filter_number*8, filter_number*16)

        self.up_1 = Up_Block(16*filter_number, 8*filter_number)
        self.up_2 = Up_Block(8*filter_number, 4*filter_number)
        self.up_3 = Up_Block(4*filter_number, 2*filter_number)
        self.up_4 = Up_Block(2*filter_number, filter_number)

        self.out = nn.Conv2d(filter_number, output_length, kernel_size=1)


    def forward(self, x):

        x = self.conv_1_1(x)
        x_1 = self.conv_1_2(x)
        x_2 = self.down_1(x_1)
        x_3 = self.down_2(x_2)
        x_4 = self.down_3(x_3)
        x_5 = self.down_4(x_4)

        x = self.up_1(x_5, x_4)
        x = self.up_2(x, x_3)
        x = self.up_3(x, x_2)
        x = self.up_4(x, x_1)

        x = self.out(x)

        return x


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

class cnn_2D(nn.Module):

    def __init__(self, input_length, output_length, filter_number):
        super(cnn_2D, self).__init__()

        self.enc1 = Conv(in_channels=input_length, out_channels=filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.enc2 = Conv(in_channels=filter_number, out_channels=filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.enc3 = Conv(in_channels=filter_number, out_channels=8*filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.enc4 = Conv(in_channels=8*filter_number, out_channels=12*filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.enc5 = Conv(in_channels=12*filter_number, out_channels=16*filter_number, kernel_size=(4, 4), stride=2, padding=1)

        self.vid1 = ConvTranspose(in_channels=16*filter_number, out_channels=16*filter_number, kernel_size=(1, 1), stride=1, padding=0)
        self.vid2 = ConvTranspose(in_channels=16*filter_number, out_channels=16*filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.vid3 = ConvTranspose(in_channels=16*filter_number, out_channels=8*filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.vid4 = ConvTranspose(in_channels=8*filter_number, out_channels=4*filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.vid5 = ConvTranspose(in_channels=4*filter_number, out_channels=24, kernel_size=(4, 4), stride=2, padding=1)
        self.vid6 = nn.ConvTranspose2d(in_channels=24, out_channels=output_length, kernel_size=(4, 4), stride=2, padding=1)


    def forward(self, x):

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        x = self.vid1(x)
        x = self.vid2(x)
        x = self.vid3(x)
        x = self.vid4(x)
        x = self.vid5(x)
        x = F.relu(self.vid6(x))

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
