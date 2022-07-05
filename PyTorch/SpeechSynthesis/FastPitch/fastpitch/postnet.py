import torch
from torch import nn
from torch.nn import functional as F

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.n_mel_channels = 80
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5
        self.postnet_n_convolutions = 5
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(self.n_mel_channels, self.postnet_embedding_dim,
                         kernel_size=self.postnet_kernel_size, stride=1,
                         padding=int((self.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(self.postnet_embedding_dim))
        )

        for i in range(1, self.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(self.postnet_embedding_dim,
                             self.postnet_embedding_dim,
                             kernel_size=self.postnet_kernel_size, stride=1,
                             padding=int(
                                 (self.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(self.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(self.postnet_embedding_dim, self.n_mel_channels,
                         kernel_size=self.postnet_kernel_size, stride=1,
                         padding=int((self.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(self.n_mel_channels))
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(
                self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        x = x.permute(0,2,1)
        return x