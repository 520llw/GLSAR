import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class FFTResNet(nn.Module):
    """FFT-enhanced ResNet Backbone"""
    
    def __init__(self, depth=50, input_channels=1, out_indices=(1, 2, 3), 
                 pretrained=True, **kwargs):
        super().__init__()
        
        if kwargs:
            print(f"🔍 FFTResNet ignored kwargs: {list(kwargs.keys())}")
        
        # Load pretrained ResNet
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
            resnet = resnet50(weights=weights)
        else:
            resnet = resnet50(weights=None)
        
        # Modify first conv for single channel
        if input_channels == 1:
            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                resnet.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.out_indices = out_indices
        
        print(f"✅ FFTResNet initialized: input_channels={input_channels}, out_indices={out_indices}")
    
    def forward(self, x):
        """Forward pass"""
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Stages
        outs = []
        x = self.layer1(x)
        if 0 in self.out_indices:
            outs.append(x)
        
        x = self.layer2(x)
        if 1 in self.out_indices:
            outs.append(x)
        
        x = self.layer3(x)
        if 2 in self.out_indices:
            outs.append(x)
        
        x = self.layer4(x)
        if 3 in self.out_indices:
            outs.append(x)
        
        return tuple(outs)