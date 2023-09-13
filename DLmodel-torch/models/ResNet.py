import torch
import torch.nn as nn
import torch.nn.functional as F
from .Layers import Conv1dSamePadding


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layers(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [3, 3, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1], 
                        kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels),
            ])

    def forward(self, x):
        y = self.layers(x)
        if self.match_channels:
            z = self.residual(x)
            y = y + z
        return y

class ResNet(nn.Module):
    def __init__(self, in_channels, mid_channels, n_class):
        super(ResNet, self).__init__()
        self.layers = nn.Sequential(*[
            ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
            ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
            ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 4),
            ResNetBlock(in_channels=mid_channels * 4, out_channels=mid_channels * 2),
        ])

        self.final = nn.Linear(mid_channels * 2, n_class)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.layers(x)
        x = self.final(x.mean(dim=-1))
        return self.softmax(x)

def test():
    x = torch.randn(2, 1, 105)
    model = ResNet(in_channels=1, mid_channels=64, n_class=2)
    y = model.forward(x)
    print(y.shape)    
    print(y)

if __name__ == '__main__':
    test()