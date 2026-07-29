import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Notes
align_corners=True when precise pixel alignment between input and output is important (e.g., segmentation tasks).
align_corners=False when general resizing is sufficient, and you're not concerned with preserving exact pixel alignment.
PPM class, since it's performing adaptive max pooling followed by interpolation, align_corners=False is likely 
the preferred option, because it is not focusing on pixel-perfect correspondence between the input and output, 
but rather on capturing multi-scale features effectively. However due to the segmentation task, i need to check and see
both options.
The original version of the PPM module only performs pooling, followed by a convolution, and then interpolation. 
It doesn’t use any normalization or activation between layers.

This version introduces more layers and functionality (GroupNorm and GELU), providing a richer feature set that could 
help the network learn more complex representations.
"""


class PPM(nn.Module):
    """
        Pyramid Pooling Module.
        Each module in the list:
          - Performs an adaptive max pooling to a smaller size.
          - Then a 1x1 convolution.
          - I added GroupNorm and GELU layers for richer feature set, needs to be verified maybe
        The outputs are upsampled back and concatenated.
    """
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ppm_modules = nn.ModuleList()

        for pool_size in self.pool_sizes:
            self.ppm_modules.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),    # nn.AdaptiveAvgPool2d(pool_size),?? AdaptiveMaxPool2d
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                    nn.GroupNorm(16, self.out_channels),
                    nn.GELU()
                )
            )

    def forward(self, x):
        ppm_outs = [F.interpolate(ppm(x), size=x.shape[2:], mode='bilinear', align_corners=True) for ppm in
                    self.ppm_modules]
        return ppm_outs


class PPMHEAD(nn.Module):
    """
    PSP (Pyramid Pooling) head used on the highest-level feature map.
    """
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=9):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=1),
            nn.GroupNorm(16, self.out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # x: N, C, H, W from the deepest feature map
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class FPNHEAD(nn.Module):
    """
    Feature Pyramid Network head to fuse multi-level features:
      - Takes multiple scales of feature maps (e.g., from backbone).
      - Uses a PPMHead on the deepest map, then merges with shallower maps.
    """
    def __init__(self, in_channels_list, out_channels=256, num_classes=9):
        super(FPNHEAD, self).__init__()
        self.num_classes = num_classes
        # PPM uses the deepest features
        self.PPMHead = PPMHEAD(in_channels=in_channels_list[-1], out_channels=out_channels, num_classes=num_classes)

        # Now we define fuse layers for the other features
        # input_fpn: [f1, f2, f3, f4] -> [64, 128, 256, 1024]
        # f4 (deepest) processed by PPM
        # fuse with f3 (256 channels), f2 (128 channels), f1 (64 channels)

        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(in_channels_list[-2], out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        # Conv_fuse2 for f2: 128 channels
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(in_channels_list[-3], out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        # Conv_fuse3 for f1: 64 channels
        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(in_channels_list[-4], out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, input_fpn):
        # in base_mixmim: L4 = 1024 x 4 x 4, L3 = 256 x 8 x 8, L2 = 128 x 16 x 16, L1 = 64 x 32 x 32
        # Adjust according to your backbone.

        # input_fpn: [f1, f2, f3, f4], matches in_channels_list order
        # f4 = deepest, f3, f2, f1 = shallower
        x1 = self.PPMHead(input_fpn[-1])  # PPM on deepest feature
        x = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)        
        # x = self.conv_x1(x) + self.Conv_fuse1(input_fpn[-2])
        x = self.conv_x1(x) + F.interpolate(self.Conv_fuse1(input_fpn[-2]), scale_factor=2, mode='bilinear', align_corners=True)
        x2 = self.Conv_fuse1_(x)

        x = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x = x +  F.interpolate(self.Conv_fuse2(input_fpn[-3]), scale_factor=2, mode='bilinear', align_corners=True)
        x3 = self.Conv_fuse2_(x)

        x = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x = x + F.interpolate(self.Conv_fuse3(input_fpn[-4]), scale_factor=2, mode='bilinear', align_corners=True)
        x4 = self.Conv_fuse3_(x)

        # Align all feature maps to x4 size and fuse
        x1 = F.interpolate(x1, x4.size()[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[2:], mode='bilinear', align_corners=True)

        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))
        return x


class ProgressiveUpsampleHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ProgressiveUpsampleHead, self).__init__()
        
        # after the FPN or final decoder, assume we have shape [N, in_channels, H/4, W/4]
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, in_channels),
            nn.GELU()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, in_channels),
            nn.GELU()
        )
        self.cls_seg = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # x is B×C×(H/4)×(W/4)
        x = self.up1(x)   # B×C×(H/2)×(W/2)
        x = self.up2(x)   # B×C×H×W
        x = self.cls_seg(x)   # B×num_classes×H×W
        return x


class UPerNet(nn.Module):
    """
    UPerNet: Unified Perceptual Parsing for Scene Understanding (Xiao et al. ECCV2018)
    This implementation expects a list of multi-level feature maps from a backbone.
    Typically: [C2, C3, C4, C5] from a ResNet or similar backbone.

    In your case, replace `self.backbone` with your MixMIM-based backbone
    and just provide the feature maps in the correct order and channels.
    """
    def __init__(self, in_channels_list, num_classes):
        super(UPerNet, self).__init__()
        self.num_classes = num_classes

        # Note: Normally, you'd define your backbone here or pass it in.
        # For example:
        # self.backbone = MyBackbone(...)

        # in_channels_list example: [64, 128, 256, 1024]
        # last element corresponds to the deepest layer

        # The FPNHEAD is set with default channels=2048 and out_channels=256.
        self.decoder = FPNHEAD(in_channels_list=in_channels_list, out_channels=256, num_classes=self.num_classes)
        self.prog_head = ProgressiveUpsampleHead(in_channels=256, num_classes=num_classes)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): List of feature maps from the backbone.
                                     Expected order: [C2, C3, C4, C5]
                                     Each element is a tensor of shape [N, C, H, W].
        Returns:
            out (Tensor): Output segmentation map of shape [N, num_classes, H, W]
        """
        x = self.decoder(features)  # B x 256 x 56 x 56 this is for mixmim_base

        x = self.prog_head(x)
        return x
