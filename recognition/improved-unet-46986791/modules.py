"""
This file defined the modules for the Improved U-Net model. It includes a main
"ImprovedUNet" class that implements the full model, in addition to various
submodules that comprise the functionality of this class.

Author: Tom Day
"""

import torch
import torch.nn as nn
        
class ContextModule(nn.Module):
    """
    Implementation of the context module from the improved unet paper [1].
    It's worth noting that this module is based upon the pre-activation
    residual block from the Identity Mappings in Deep Residual Networks paper [4]
    """
    def __init__(self, in_channels):
        """
        Instanciates the ContextModule class.

        Args:
            in_channels: Number of input and output channels.

        returns:
            ContextModule
        """
        super().__init__()

        # Batchnorm replaced by instance norm in the paper for small batch sizes
        # Since in our implementation we will be using a larger batch size, it
        # may make sense to change this back to a batchnorm as per [4]?
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout2d(p=0.3)

        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass of the context module
        Args:
            x: input tensor

        Returns:
            A tensor of the same shape.
        """
        residual = x

        # conv block 1
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)

        # Dropout between blocks
        x = self.dropout(x)
        
        # conv block 2
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)

        # add residual back onto the output
        x = x + residual
        
        return x

class DownsamplingModule(nn.Module):
    """
    Downsampling Module.
    Reduces the spatial resolution and increases the number of features.
    """
    def __init__(self, in_channels, out_channels):
        """
        Instanciates the DownSampling module.

        Args:
            in_channels: The number of input channels
            out_channels: The number of output channels

        returns:
            DownsamplingModule
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        """
        Forward pass of the downsampling module.
        Args:
            x: The tensor to apply the downsampling module to
        returns:
            A tensor that has been downsampled
        """
        x = self.conv(x)
        x = self.act(x)

        return x


class UpsamplingModule(nn.Module):
    """
    Upsampling module from the Improved U-Net paper.
    Upsamples the input by a factor of two while halving the number of
    channels/features. Used in place of transposed convs in improved Unet.
    """
    def __init__(self, in_channels):
        """
        Instanciates an upsamplign module
        args:
            in_channels: The number of input channels

        returns:
            UpsamplingModule
        """
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        """
        Forward pass of the Upsampling Module.
        args:
            x: Input tensor to apply forward pass to

        returns:
            A tensor which has been upsampled
        """
        # Upsample input
        x = self.upsample(x)
        # Half the channels
        x = self.conv(x)
        x = self.act(x)

        return x
    

class LocalisationModule(nn.Module):
    """
    Localisation module from the Improved U-Net paper

    A 3x3 convolution followed by a 1x1 convolution that halves the number of
    channels.
    """
    def __init__(self, in_channels):
        """
        Instanciates a Localisation Module
        Args:
            in_channels: The number of input channels.
        Returns:
            LocalisationModule
        """
        
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        """
        Forward pass of the localisation module
        args:
            x: The tensor to apply the forward pass to
        returns:
            The output tensor from the localisation module
        """
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        return x

class SegmentationLayer(nn.Module):
    """
    1x1 conv for the paper's "deep supervisionn" mechanism
    Dark green blocks on the diagram
    """
    def __init__(self, in_channels, seg_class_count):
        """
        Instanciates a SegmentationLayer
        args:
            in_channels: The number of input channels
            seg_class_count: The number of classes being segmented
        Returns:
            SegmentationLayer
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, seg_class_count, kernel_size=1)

    def forward(self, x):
        """
        A forward pass of the Segmentation layr
        Args:
            x: input tensor
        Returns:
            output tensor
        """
        x = self.conv(x)
        
        return x
        

class ImprovedUNet(nn.Module):
    """
    Implementation of the improved unet model architecture [1] in two dimensions.
    Consists of an encoder-decoder network with skip connections
    """
    def __init__(self):
        """
        Constructs an ImprovedUnet
        returns:
            ImprovedUnet
        """
        super().__init__()
        self.class_count = 6

        filter_sizes = [16, 32, 64, 128, 256]
        
        # layer naming convention: `{module name}_{level}` where 0 is the first level
        # Levels are based on where the layer is in the diagram (Improved unet paper, fig. 1).

        # Encoder
        # Initial conv moves input into feature space
        self.initial_conv = nn.Conv2d(1, filter_sizes[0], kernel_size=3, padding=1)
        self.context_0 = ContextModule(filter_sizes[0])

        self.downsample_1 = DownsamplingModule(filter_sizes[0], filter_sizes[1])
        self.context_1 = ContextModule(filter_sizes[1])
        
        self.downsample_2 = DownsamplingModule(filter_sizes[1], filter_sizes[2])
        self.context_2 = ContextModule(filter_sizes[2])

        self.downsample_3 = DownsamplingModule(filter_sizes[2], filter_sizes[3])
        self.context_3 = ContextModule(filter_sizes[3])

        self.downsample_4 = DownsamplingModule(filter_sizes[3], filter_sizes[4])
        
        # Bottleneck
        self.context_4 = ContextModule(filter_sizes[4])

        # Decoder
        self.upsample_4 = UpsamplingModule(filter_sizes[4])

        self.localisation_3 = LocalisationModule(filter_sizes[3] * 2)
        self.upsample_3 = UpsamplingModule(filter_sizes[3])

        self.localisation_2 = LocalisationModule(filter_sizes[2] * 2)
        self.seg_2 = SegmentationLayer(filter_sizes[2], self.class_count)
        self.upsample_2 = UpsamplingModule(filter_sizes[2])

        self.localisation_1 = LocalisationModule(filter_sizes[1] * 2)
        self.seg_1 = SegmentationLayer(filter_sizes[1], self.class_count)
        self.upsample_1 = UpsamplingModule(filter_sizes[1])

        self.final_conv = nn.Conv2d(filter_sizes[0] * 2, filter_sizes[1], kernel_size=3, padding=1)
        self.seg_0 = SegmentationLayer(filter_sizes[1], self.class_count)

        # Deep supervision upscaling - feature maps from different "levels" have
        # different resolutions. As per the paper we must upscale them to be
        # combined with the final model output.
        self.upscale_seg_2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscale_seg_1 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the improved U-Net model.
        Args:
            x: Input image tensor of shape (Batch, 1, H, W)

        Returns:
            A tensor containig a class probability map in the shape
            (Batch, 6, H, W)
        """

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
        # Bottleneck
        x = self.context_4(x)

        # Decoder
        x = self.upsample_4(x)
        
        x = torch.cat([x, skip3], dim=1)

        x = self.localisation_3(x)
        x = self.upsample_3(x)

        x = torch.cat([x, skip2], dim=1)
        x = self.localisation_2(x)
        seg_lvl_2 = self.seg_2(x)
        x = self.upsample_2(x)

        x = torch.cat([x, skip1], dim=1)
        x = self.localisation_1(x)
        seg_lvl_1 = self.seg_1(x)
        x = self.upsample_1(x)

        x = torch.cat([x, skip0], dim=1)
        x = self.final_conv(x)
        seg_lvl_0 = self.seg_0(x)

        # deep supervision
        # First upsample the intermediate logits
        upscale_lvl_2 = self.upscale_seg_2(seg_lvl_2)
        upscale_lvl_1 = self.upscale_seg_1(seg_lvl_1)

        # Then add the upscaled logits and the final model output
        out = upscale_lvl_2 + upscale_lvl_1
        out = out + seg_lvl_0

        # Softmax
        out = self.output(out)
        
        return out
