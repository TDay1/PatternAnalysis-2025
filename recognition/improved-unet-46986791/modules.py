import torch.nn as nn
        
class ContextModule(nn.Module):
    """
    implementation of the context module from the improved unet paper (which is a modified residual layer from the Identity Mappings in Deep Residual Networks paper)
    TODO: cite
    """
    def __init__(self, in_channels):
        super().__init__()

        # Batchnorm replaced by instance norm in the paper for small batch sizes
        # TODO: since we will have larger batch sizes (we training in 2d) it may make sense to switch back to batchnorm??
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3)

        self.dropout = nn.Dropout2d(p=0.3)

        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.dropout(x)
        
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)

        x = x + residual
        
        return x

class DownsamplingModule(nn.Module):
    """
    Down sample (the gold blocks in the paper's diagram)
    Just a 3x3 conv with 2 stride
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x