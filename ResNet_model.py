import torch
import torch.nn as nn
import math

# to check the model parameter count
from torchvision.models import resnet50

# We are starting with ResNet 50 model then make changes to it as explained in paper for ConvNexT
# stem block
class StemBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.stem(x)
        return x
        

# Bottleneck block
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsampling=1, expansion=4):
        super().__init__()
        self.expanded_channels = out_channels*expansion
        self.need_shortcut = in_channels != self.expanded_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=downsampling, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, self.expanded_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.expanded_channels)
        )
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, self.expanded_channels, kernel_size=1, stride=downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)
        ) if self.need_shortcut else nn.Identity()

    def forward(self, x):
        residual = x
        if self.need_shortcut: residual = self.shortcut(x)
        x = self.block(x)
        x += residual
        x = nn.ReLU(x)
        return x

# bottle = BottleneckBlock(32, 64)
# bottle(torch.ones((1, 32, 10, 10))).shape
# print(bottle)

# ResNet Stage
class ResNetStage(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, expansion=4):
        super().__init__()
        downsampling = 2 if (in_channels != out_channels) else 1
        self.stage = nn.Sequential(
            BottleneckBlock(in_channels, out_channels, downsampling, expansion),
            *[BottleneckBlock(out_channels*expansion, out_channels, downsampling=1) for _ in range(n-1)]
        )

    def forward(self, x):
        x = self.stage(x)
        return x

# dummy = torch.ones((1, 32, 48, 48))

# bottle = ResNetStage(64, 128, n=3)
# # bottle(dummy)
# print(bottle)

# Classification Head
class ResNetClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        x = self.head(x)
        return x

# ResNet Model - the initial values are for ResNet50 model
class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, expansion=4, block_sizes=[64, 128, 256, 512], depths=[3, 4, 6, 3]):
        super().__init__()

        self.in_out = list(zip(block_sizes, block_sizes[1:]))

        self.stem = StemBlock(in_channels, block_sizes[0])
        self.mainblock = nn.Sequential(
            ResNetStage(block_sizes[0], block_sizes[0], depths[0]),
            *[ResNetStage(in_channel*expansion, out_channel, depth) for ((in_channel, out_channel), depth) in zip(self.in_out, depths[1:])]
        )
        self.head = ResNetClassificationHead(block_sizes[-1]*expansion, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.mainblock(x)
        x = self.head(x)
        return x
    
dummy = torch.ones((1, 32, 48, 48))

bottle = ResNet(3, 1000)
# bottle(dummy)
print(bottle)

print(sum(p.numel() for p in bottle.parameters()))

print(sum(p.numel() for p in resnet50().parameters()))
