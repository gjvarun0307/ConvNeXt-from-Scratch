import torch
import torch.nn as nn
import math
from torchvision.ops import StochasticDepth

# Changes (Micro Changes) - changed ReLU to GELU and BatchNorm to LayerNorm to whole code below
class ConvnextLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape, eps)

    def forward(self, x):
        x = torch.permute(x, (0,2,3,1))  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = torch.permute(x, (0,3,1,2))  # (N, H, W, C) -> (N, C, H, W)
        return x
    
# Changes (Micro Chnages) - add saperate downsampling blocks
class ConvNextDownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsampling = nn.Sequential(
            ConvnextLayerNorm(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.downsampling(x)
        return x


# We are starting with ResNet 50 model then make changes to it as explained in paper for ConvNexT
# Changed some class names for ConvNext
# Changes (Macro Design - "Patchify" stem block): Change the kernel and stride to 4
# stem block
class StemBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4),
            ConvnextLayerNorm(out_channels),
        )

    def forward(self, x):
        x = self.stem(x)
        return x
        

# Bottleneck block
# Changes (ResNext-ify): Changed the spatial convolution from standard to depthwise convolution
# Changes (Inverted Bottleneck): moved the spatial convolution to top adn invert dimensions; from wise -> narrow -> wide to narrow -> wide -> narrow
# Chnages (Using Large Kernal sizes): 
class BottleneckBlock(nn.Module):
    def __init__(self, channels, expansion=4, drop_path=0.):
        super().__init__()
        self.expanded_channels = channels*expansion
        self.gamma = nn.Parameter(1e-6 * torch.ones(channels), requires_grad=True)

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, stride=1, groups=channels),
            ConvnextLayerNorm(channels),
            nn.Conv2d(channels, self.expanded_channels, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(self.expanded_channels, channels, kernel_size=1, stride=1),
        )
        
        self.drop_path = StochasticDepth(drop_path, mode="batch")

    def forward(self, x):
        residual = x
        x = self.block(x)
        x = self.gamma.view(1, -1, 1, 1) * x
        x = residual + self.drop_path(x)
        return x

# ConvNext Stage
class ConvNextStage(nn.Module):
    def __init__(self, channels, drop_path, n=1):
        super().__init__()
        self.stage = nn.Sequential(
            *[BottleneckBlock(channels, drop_path=drop_path[i]) for i in range(n)]
        )

    def forward(self, x):
        x = self.stage(x)
        return x

# Classification Head
class ConvNextClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        x = self.head(x)
        return x

# ConvNext Model 
# Changes (Macro Design - Stage Ratio Changes): from [3, 4, 6, 3] to [3, 3, 9, 3]
# Changes (ResNext-ify): Change the block sizes to support increase in width sizes
# Changes (Micro Changes): Adding the saparate downsample blocks
class ConvNext(nn.Module):
    def __init__(self, in_channels, num_classes, block_sizes=[96, 192, 384, 768], depths=[3, 3, 9, 3], drop_path_rate=0.):
        super().__init__()

        self.in_out = list(zip(block_sizes, block_sizes[1:]))
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = depths[0]

        self.stem = StemBlock(in_channels, block_sizes[0])
        self.mainblock = nn.Sequential(ConvNextStage(block_sizes[0], dp_rates[0:depths[0]], depths[0]))
        for i in range(3):
            self.mainblock.append(ConvNextDownsamplingBlock(block_sizes[i], block_sizes[i+1]))
            self.mainblock.append(ConvNextStage(block_sizes[i+1], dp_rates[cur:cur+depths[i+1]], depths[i+1]))
            cur += depths[i+1]
        
        self.head = ConvNextClassificationHead(block_sizes[-1], num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.mainblock(x)
        x = self.head(x)
        return x

def get_model(config):
    model = ConvNext(in_channels=config["in_channels"], num_classes=config["num_classes"], block_sizes=config["block_sizes"], depths=config["depths"], drop_path_rate=config["drop_path_rate"])
    return model

# model = ConvNext(3, 1000)
# print(model)
# print(sum(p.numel() for p in model.parameters()))

# from torchvision.models import convnext_tiny
# print(sum(p.numel() for p in convnext_tiny().parameters()))
