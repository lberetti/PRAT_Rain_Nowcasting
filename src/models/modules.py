import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajGRU_cell(nn.Module):

    def __init__(self, input_size, hidden_filters, L, sequence_length, device):

        super(TrajGRU_cell, self).__init__()

        self.device = device

        self.sequence_length = sequence_length
        self.hidden_filters = hidden_filters

        # Input size + padding
        self.state_height = input_size[1]
        self.state_width = input_size[2]

        input_channels = input_size[0]
        # Input to hidden conv Layer (contains the reset gate, the update gate and the new information)
        self.i2h = nn.Conv2d(in_channels=input_channels,
                            out_channels=3*hidden_filters,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1))

        # Convolutional layers of the flow generator network lambda in the paper.
        self.i2f = nn.Conv2d(in_channels=input_channels,
                            out_channels=32,
                            kernel_size=(5, 5),
                            stride=(1, 1),
                            padding=(2, 2))

        self.h2f = nn.Conv2d(in_channels=hidden_filters,
                            out_channels=32,
                            kernel_size=(5, 5),
                            stride=(1, 1),
                            padding=(2, 2))

        self.flows = nn.Conv2d(in_channels=32,
                              out_channels=L*2,
                              kernel_size=(5, 5),
                              stride=(1, 1),
                              padding=(2, 2))

        # Convolutional layer for the warp data convolution (weights for projecting the channels)
        self.wrap_conv = nn.Conv2d(in_channels=hidden_filters*L,
                                  out_channels=hidden_filters*3,
                                  kernel_size=(1, 1),
                                  stride=(1, 1))


    def flow_generator(self, inputs, states):
        if inputs is not None:
            i2f_output = self.i2f(inputs)
        else:
            i2f_output = None
        h2f_output = self.h2f(states.to(self.device))

        if inputs is not None:
            flows_output = torch.tanh(i2f_output + h2f_output)
        else:
            flows_output = torch.tanh(h2f_output)

        flows_output = self.flows(flows_output)
        flows_output = torch.split(flows_output, 2, dim=1)

        return flows_output


    def wrap(self, input, flow):
        batch_size, channels, height, width = input.size()
        # mesh grid
        x = torch.arange(0, width).view(1, -1).repeat(height, 1)
        y = torch.arange(0, height).view(-1, 1).repeat(1, width)
        x = x.view(1, 1, height, width).repeat(batch_size, 1, 1, 1)
        y = y.view(1, 1, height, width).repeat(batch_size, 1, 1, 1)
        grid = torch.cat((x, y), 1).float()
        vgrid = grid.to(self.device) + flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / width - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / height - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(input.to(self.device), vgrid)
        return output


    def forward(self, inputs, state):

        # At the first iteration
        if state is None:
            state = torch.zeros((inputs.size(0), self.hidden_filters, self.state_height, self.state_width)).to(self.device)

        if inputs is not None:
            batch_size, sequence_length, channels, height, width = inputs.size()
            i2h_output = self.i2h(torch.reshape(inputs, (-1, channels, height, width)))
            i2h_output = torch.reshape(i2h_output, (batch_size, sequence_length, i2h_output.size(1), i2h_output.size(2), i2h_output.size(3)))
            reset_gate_i2h, update_gate_i2h, new_info_gate_i2h = torch.split(i2h_output, self.hidden_filters, dim=2)
        else:
            reset_gate_i2h, update_gate_i2h, new_info_gate_i2h = None, None, None

        previous_h = state
        outputs = []

        for k in range(self.sequence_length):
            if inputs is not None:
                flows_o = self.flow_generator(inputs[:, k, ...], previous_h)
            else:
                flows_o = self.flow_generator(None, previous_h)

            wrapped_data = []
            for i in range(len(flows_o)):
                wrapped_data.append(self.wrap(previous_h, -flows_o[i]))
            wrapped_data = torch.cat(wrapped_data, dim=1)
            h2h_output = self.wrap_conv(wrapped_data)

            reset_gate_h2h, update_gate_h2h, new_info_gate_h2h = torch.split(h2h_output, self.hidden_filters, dim=1)

            if reset_gate_i2h is not None:
                reset_gate = torch.sigmoid(reset_gate_i2h[:, k, ...] + reset_gate_h2h)
            else:
                reset_gate = torch.sigmoid(reset_gate_h2h)

            if update_gate_i2h is not None:
                update_gate = torch.sigmoid(update_gate_i2h[:, k, ...] + update_gate_h2h)
            else:
                update_gate = torch.sigmoid(update_gate_h2h)

            if new_info_gate_i2h is not None:
                new_info_gate = torch.sigmoid(new_info_gate_i2h[:, k, ...] + reset_gate*new_info_gate_h2h)
            else:
                new_info_gate = torch.sigmoid(reset_gate*new_info_gate_h2h)

            h = (1 - update_gate) * new_info_gate + update_gate * previous_h
            outputs.append(h)
            previous_h = h

        outputs = torch.stack(outputs, dim=1)

        return outputs, h




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
