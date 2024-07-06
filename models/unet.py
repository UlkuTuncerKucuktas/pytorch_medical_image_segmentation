import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetEncoder(nn.Module):
    def __init__(self, input_channels, channel_sizes):
        super(UNetEncoder, self).__init__()
        self.encoders = nn.ModuleList()
        in_channels = input_channels
        for out_channels in channel_sizes:
            self.encoders.append(self._block(in_channels, out_channels))
            in_channels = out_channels

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        skip_connections = []
        for encode in self.encoders:
            x = encode(x)
            skip_connections.append(x)
        return x, skip_connections[:-1]

class UNetDecoder(nn.Module):
    def __init__(self, num_classes, channel_sizes, output_activation=nn.Sigmoid()):
        super(UNetDecoder, self).__init__()
        self.decoders = nn.ModuleList()
        self.output_activation = output_activation
        reversed_sizes = list(reversed(channel_sizes))
        for i in range(len(reversed_sizes) - 1):
            self.decoders.append(self._block(reversed_sizes[i] * 2, reversed_sizes[i + 1]))
        self.final_conv = nn.Conv2d(reversed_sizes[-1], num_classes, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connections):
        for i, decode in enumerate(self.decoders):
            if i < len(skip_connections):
                x = torch.cat((x, skip_connections[-(i+1)]), dim=1)
            x = decode(x)
        x = self.final_conv(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x
