import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn_2D(nn.Module):

    def __init__(self, input_length, output_length, filter_number):
        super(cnn_2D, self).__init__()

        self.enc1 = nn.Conv2d(in_channels=input_length, out_channels=filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.enc2 = nn.Conv2d(in_channels=filter_number, out_channels=filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.enc3 = nn.Conv2d(in_channels=filter_number, out_channels=8*filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.enc4 = nn.Conv2d(in_channels=8*filter_number, out_channels=12*filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.enc5 = nn.Conv2d(in_channels=12*filter_number, out_channels=16*filter_number, kernel_size=(4, 4), stride=2, padding=1)

        self.vid1 = nn.ConvTranspose2d(in_channels=16*filter_number, out_channels=16*filter_number, kernel_size=(1, 1), stride=1, padding=0)
        self.vid2 = nn.ConvTranspose2d(in_channels=16*filter_number, out_channels=16*filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.vid3 = nn.ConvTranspose2d(in_channels=16*filter_number, out_channels=8*filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.vid4 = nn.ConvTranspose2d(in_channels=8*filter_number, out_channels=4*filter_number, kernel_size=(4, 4), stride=2, padding=1)
        self.vid5 = nn.ConvTranspose2d(in_channels=4*filter_number, out_channels=24, kernel_size=(4, 4), stride=2, padding=1)
        self.vid6 = nn.ConvTranspose2d(in_channels=24, out_channels=output_length, kernel_size=(4, 4), stride=2, padding=1)


    def forward(self, x):

        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        x = F.relu(self.vid1(x))
        x = F.relu(self.vid2(x))
        x = F.relu(self.vid3(x))
        x = F.relu(self.vid4(x))
        x = F.relu(self.vid5(x))
        x = F.relu(self.vid6(x))

        return x
