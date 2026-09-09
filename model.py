import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU, twice, then spatial downsampling by 2."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    """
    Small CNN encoder for CIFAR-10 (3x32x32).
    Ends with global average pooling, so the output is a fixed-size
    (batch, out_dim) vector regardless of the spatial resolution reached
    by the conv blocks.
    """

    def __init__(self, in_channels=3, channels=(64, 128, 256)):
        super().__init__()
        blocks = []
        prev = in_channels
        for c in channels:
            blocks.append(ConvBlock(prev, c))
            prev = c
        self.blocks  = nn.Sequential(*blocks)
        self.out_dim = channels[-1]

    def forward(self, x):
        x = self.blocks(x)           # (batch, out_dim, H', W')
        x = x.mean(dim=[2, 3])       # global average pooling -> (batch, out_dim)
        return x


class Expander(nn.Module):
    """
    Projects the representation into a higher-dimensional space where the
    VICReg loss is computed. Two (Linear -> BN -> ReLU) layers, then a final
    plain Linear layer (no activation on the output).
    """

    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)
