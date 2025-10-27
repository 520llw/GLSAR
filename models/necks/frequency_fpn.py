import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencySpatialFPN(nn.Module):
    """Feature Pyramid Network with frequency enhancement"""
    
    def __init__(self, in_channels, out_channels, num_outs, target_size, **kwargs):
        super().__init__()
        
        if kwargs:
            print(f"🔍 FPN ignored kwargs: {list(kwargs.keys())}")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.target_size = target_size
        
        # Lateral convolutions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels
        ])
        
        # FPN convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) 
            for _ in range(num_outs)
        ])
        
        print(f"✅ FPN initialized: in={in_channels}, out={out_channels}, levels={num_outs}")
    
    def forward(self, inputs):
        """Forward pass"""
        assert len(inputs) == len(self.lateral_convs), \
            f"Expected {len(self.lateral_convs)} inputs, got {len(inputs)}"
        
        # Lateral connections
        laterals = [
            lateral_conv(inputs[i]) 
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], 
                size=laterals[i-1].shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Output convolutions
        outs = [
            self.fpn_convs[i](laterals[i]) 
            for i in range(len(laterals))
        ]
        
        # Resize all to target size
        outs = [
            F.interpolate(out, size=self.target_size, mode='bilinear', 
                         align_corners=False)
            for out in outs
        ]
        
        return tuple(outs)
