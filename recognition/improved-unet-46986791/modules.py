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
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout2d(p=0.3)

        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

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

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x


class UpsamplingModule(nn.Module):
    """
    From the paper: "first upsampling the low resolution feature maps, which is done by means of a simple upscale that repeats the feature voxels twice in each spatial dimension, followed by a 3x3x3 convolution that halves the number of feature maps" [CITE]
    It is the blue blocks in fig. 1 from the improved unet paper.
    TODO: unsure how well this will work compared to transposed convs. change this if bad results.
    """
    def __init__(self, in_channels):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.act(x)

        return x
        

class LocalisationModule(nn.Module):
    """
    implementation of the localisation module from the paper (fig. 1 orange blocks).
    From the paper: "A localization module consists of a 3x3x3 convolution followed by a 1x1x1 convolution that halves the number of feature maps" [cite]
    """
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        return x


class ImprovedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classes = 6

        filter_sizes = [16, 32, 64, 128, 256]
        
        # layer naming convention: `{module name}_{level}` where 0 is the first level
        # Levels are based on where the layer is in the diagram (Improved unet paper, fig. 1).

        # Encoder
        self.initial_conv = nn.Conv2d(1, filter_sizes[0], kernel_size=3, padding=1)
        self.context_0 = ContextModule(filter_sizes[0])

        self.downsample_1 = DownsamplingModule(filter_sizes[0], filter_sizes[1])
        self.context_1 = ContextModule(filter_sizes[1])
        
        self.downsample_2 = DownsamplingModule(filter_sizes[1], filter_sizes[2])
        self.context_2 = ContextModule(filter_sizes[2])

        self.downsample_3 = DownsamplingModule(filter_sizes[2], filter_sizes[3])
        self.context_3 = ContextModule(filter_sizes[3])

        self.downsample_4 = DownsamplingModule(filter_sizes[3], filter_sizes[4])
        self.context_4 = ContextModule(filter_sizes[4])

    def forward(self, x):

        # encoder
        x = self.initial_conv(x)
        skip0 = self.context_0(x)

        x = self.downsample_1(skip0)
        skip1 = self.context_1(x)

        x = self.downsample_2(skip1)
        skip2 = self.context_2(x)

        x = self.downsample_3(skip2)
        skip3 = self.context_3(x)

        x = self.downsample_4(skip3)
        x = self.context_4(x)

        return x