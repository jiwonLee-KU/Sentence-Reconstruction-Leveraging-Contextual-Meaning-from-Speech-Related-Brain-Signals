import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        # --------------------- Dilated Causal Convolution ---------------------
        print("padding", padding)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        # ----------------------------------------------------------------------
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # ----------------------------------------------------------------------


        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        # ----------------------------------------------------------------------
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # Residual Connections
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.HE_init()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    def HE_init(self):
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_uniform_(self.downsample.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalBlock_end(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock_end, self).__init__()
        # --------------------- Dilated Causal Convolution ---------------------
        print("dropout", dropout)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        # ----------------------------------------------------------------------
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # --------------------- Dilated Causal Convolution ---------------------
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        # ----------------------------------------------------------------------
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1,self.relu1,self.dropout1,
                                 self.conv2, self.chomp2, self.dropout2)

        # Residual Connections
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]

            out_channels = num_channels[i]

            if i < num_levels - 1:
                print(i)
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            elif i == num_levels-1:
                layers += [TemporalBlock_end(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout)]




        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Generator(nn.Module):
    def __init__(self, input_size, ndim, num_channels, kernel_size, dropout):
        super(Generator, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.adaptivepool_1d = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc_ap = nn.Linear(num_channels[len(num_channels)-1], num_channels[len(num_channels)-1]//2)
        self.fc_ap2 = nn.Linear(num_channels[len(num_channels) - 1] // 2, ndim)
        self.fc_layer = nn.Sequential(self.fc_ap,self.fc_ap2)

    def forward(self, inputs):
        y1 = self.tcn(inputs)
        y1 = self.adaptivepool_1d(y1)
        # y1 = nn.ReLU()(y1)
        y1 = self.flatten(y1)
        output = self.fc_layer(y1)


        return output



class Vec2onehot_Discriminator(nn.Module):
    def __init__(self):
        super(Vec2onehot_Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(20, 32),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        # print('x shape', self.model(x).shape)
        validity = self.discriminator(x)
        return validity

