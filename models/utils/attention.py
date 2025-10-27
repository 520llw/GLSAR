"""
Attention Modules for SAR Ship Detection
专为SAR图像设计的注意力机制模块（添加 target_size 支持）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    结合通道注意力和空间注意力
    
    Reference: CBAM: Convolutional Block Attention Module (ECCV 2018)
    """
    
    # 1. 新增 target_size 参数，默认值与配置一致
    def __init__(self, channels, reduction=16, kernel_size=7, target_size=None):
        super().__init__()
        self.channels = channels
        # 保存 target_size，默认512x512
        self.target_size = target_size if target_size is not None else (512, 512)
        print(f"✅ CBAM initialized: channels={channels}, target_size={self.target_size}")
        
        # ========== Channel Attention ==========
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # ========== Spatial Attention ==========
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, 
                                      padding=kernel_size // 2, 
                                      bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        identity = x
        
        # 2. 适配 target_size（确保输入尺寸统一）
        if (x.shape[2], x.shape[3]) != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        # ========== Channel Attention ==========
        avg_out = self.channel_fc(self.avg_pool(x))
        max_out = self.channel_fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # ========== Spatial Attention ==========
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_concat))
        out = x * spatial_att
        
        # 3. 恢复原始尺寸（若输入被调整过）
        if out.shape[2:] != identity.shape[2:]:
            out = F.interpolate(out, size=identity.shape[2:], mode='bilinear', align_corners=False)
        
        return out


class SARScatteringAttention(nn.Module):
    """
    SAR Scattering Attention
    专门处理SAR图像中的散射特性
    
    SAR图像的散射强度包含重要的目标信息
    """
    
    # 新增 target_size 参数
    def __init__(self, channels, reduction=8, target_size=None):
        super().__init__()
        self.channels = channels
        self.target_size = target_size if target_size is not None else (512, 512)
        print(f"✅ SARScatteringAttention initialized: channels={channels}, target_size={self.target_size}")
        
        # 散射强度特征提取
        self.scatter_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        identity = x
        
        # 适配 target_size
        if (x.shape[2], x.shape[3]) != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        # 提取散射特征
        scatter_feat = self.scatter_conv(x)
        
        # 门控调制
        gate_weight = self.gate(scatter_feat)
        
        # 加权融合
        out = x + scatter_feat * gate_weight
        
        # 恢复原始尺寸
        if out.shape[2:] != identity.shape[2:]:
            out = F.interpolate(out, size=identity.shape[2:], mode='bilinear', align_corners=False)
        
        return out


class SpeckleNoiseAttention(nn.Module):
    """
    Speckle Noise Attention
    专门处理SAR图像中的斑点噪声
    
    斑点噪声是SAR图像的固有特性，需要自适应抑制
    """
    
    # 新增 target_size 参数
    def __init__(self, channels, kernel_size=3, target_size=None):
        super().__init__()
        self.channels = channels
        self.target_size = target_size if target_size is not None else (512, 512)
        print(f"✅ SpeckleNoiseAttention initialized: channels={channels}, target_size={self.target_size}")
        
        # 噪声估计网络
        self.noise_estimator = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 
                     padding=kernel_size // 2, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        # 特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        identity = x
        
        # 适配 target_size（保证噪声估计的一致性）
        if (x.shape[2], x.shape[3]) != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        # 估计噪声分布
        noise_map = self.noise_estimator(x)
        
        # 自适应抑制噪声
        clean_feat = x * (1 - noise_map)
        
        # 特征增强
        out = self.enhance(clean_feat) + x
        
        # 恢复原始尺寸
        if out.shape[2:] != identity.shape[2:]:
            out = F.interpolate(out, size=identity.shape[2:], mode='bilinear', align_corners=False)
        
        return out


class CrossLevelAttention(nn.Module):
    """
    Cross-Level Attention
    用于FPN不同层级之间的特征交互（重点适配 target_size）
    """
    
    # 新增 target_size 参数，指定跨层对齐的基准尺寸
    def __init__(self, channels, num_levels=3, target_size=None):
        super().__init__()
        self.channels = channels
        self.num_levels = num_levels
        # 4. 用 target_size 替代原有的“以第1层为基准”，统一跨层对齐尺寸
        self.target_size = target_size if target_size is not None else (512, 512)
        print(f"✅ CrossLevelAttention initialized: channels={channels}, levels={num_levels}, target_size={self.target_size}")
        
        # 跨层级特征融合
        self.cross_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 1) 
            for _ in range(num_levels)
        ])
        
        # 注意力权重生成
        self.att_conv = nn.Sequential(
            nn.Conv2d(channels * num_levels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_levels, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, feats):
        """
        Args:
            feats: list of [B, C, H, W] tensors
        Returns:
            out: list of [B, C, H, W] tensors
        """
        assert len(feats) == self.num_levels
        
        # 5. 统一尺寸到 target_size（替代原有的“以第1层为基准”，更灵活）
        aligned_feats = []
        for i, feat in enumerate(feats):
            # 先卷积调整通道，再对齐到 target_size
            conv_feat = self.cross_convs[i](feat)
            aligned_feat = F.interpolate(conv_feat, size=self.target_size, 
                                        mode='bilinear', align_corners=False)
            aligned_feats.append(aligned_feat)
        
        # 拼接所有层级
        concat_feats = torch.cat(aligned_feats, dim=1)
        
        # 生成注意力权重（基于 target_size 的统一特征图）
        att_weights = self.att_conv(concat_feats)  # [B, num_levels, H_target, W_target]
        
        # 加权融合并恢复原始尺寸
        outputs = []
        for i, (feat, aligned_feat) in enumerate(zip(feats, aligned_feats)):
            weight = att_weights[:, i:i+1, :, :]  # [B, 1, H_target, W_target]
            enhanced_aligned = aligned_feat * (1 + weight)  # 加权增强
            
            # 恢复到当前层级的原始尺寸
            enhanced = F.interpolate(enhanced_aligned, size=feat.shape[2:],
                                    mode='bilinear', align_corners=False)
            outputs.append(enhanced)
        
        return outputs


class TargetAwareAttention(nn.Module):
    """
    Target-Aware Attention
    根据目标特征动态调整注意力
    """
    
    # 新增 target_size 参数
    def __init__(self, channels, num_queries=100, target_size=None):
        super().__init__()
        self.channels = channels
        self.num_queries = num_queries
        self.target_size = target_size if target_size is not None else (512, 512)
        print(f"✅ TargetAwareAttention initialized: channels={channels}, queries={num_queries}, target_size={self.target_size}")
        
        # Query生成
        self.query_embed = nn.Embedding(num_queries, channels)
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        identity = x
        B, C, H, W = x.shape
        
        # 适配 target_size（确保query注意力计算的一致性）
        if (H, W) != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
            H, W = self.target_size  # 更新尺寸
        
        # Flatten spatial dimensions
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Generate queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, C]
        
        # Multi-head attention
        attn_out, _ = self.multihead_attn(
            query=queries,
            key=x_flat,
            value=x_flat
        )  # [B, num_queries, C]
        
        # 聚合query features
        global_feat = attn_out.mean(dim=1, keepdim=True)  # [B, 1, C]
        global_feat = global_feat.permute(0, 2, 1).view(B, C, 1, 1)  # [B, C, 1, 1]
        
        # 调制原始特征
        out = x * torch.sigmoid(global_feat)
        out = self.out_proj(out)
        
        # 恢复原始尺寸
        if out.shape[2:] != identity.shape[2:]:
            out = F.interpolate(out, size=identity.shape[2:], mode='bilinear', align_corners=False)
        
        return out


# ========== 测试代码（更新为支持 target_size 的测试） ==========
if __name__ == '__main__':
    print("Testing Attention Modules (with target_size support)...")
    
    # 测试参数（指定 target_size=256x256）
    B, C, H, W = 2, 256, 128, 128  # 输入尺寸小于 target_size
    target_size = (256, 256)
    x = torch.randn(B, C, H, W)
    
    # 测试CBAM
    print("\n1. Testing CBAM...")
    cbam = CBAM(C, reduction=16, target_size=target_size)
    out = cbam(x)
    assert out.shape == x.shape  # 输出恢复为原始输入尺寸
    print(f"   Input: {x.shape} -> Output: {out.shape} ✅")
    
    # 测试SARScatteringAttention
    print("\n2. Testing SARScatteringAttention...")
    scatter_att = SARScatteringAttention(C, target_size=target_size)
    out = scatter_att(x)
    assert out.shape == x.shape
    print(f"   Input: {x.shape} -> Output: {out.shape} ✅")
    
    # 测试SpeckleNoiseAttention
    print("\n3. Testing SpeckleNoiseAttention...")
    speckle_att = SpeckleNoiseAttention(C, target_size=target_size)
    out = speckle_att(x)
    assert out.shape == x.shape
    print(f"   Input: {x.shape} -> Output: {out.shape} ✅")
    
    # 测试CrossLevelAttention（重点验证 target_size 对齐）
    print("\n4. Testing CrossLevelAttention...")
    feats = [
        torch.randn(B, C, 64, 64),   # 原始尺寸1
        torch.randn(B, C, 32, 32),   # 原始尺寸2
        torch.randn(B, C, 16, 16)    # 原始尺寸3
    ]
    cross_att = CrossLevelAttention(C, num_levels=3, target_size=target_size)
    out_feats = cross_att(feats)
    assert len(out_feats) == 3
    for i, (f_in, f_out) in enumerate(zip(feats, out_feats)):
        assert f_out.shape == f_in.shape  # 输出恢复为各层原始尺寸
        print(f"   Level {i}: {f_in.shape} -> {f_out.shape} ✅")
    
    # 测试TargetAwareAttention
    print("\n5. Testing TargetAwareAttention...")
    target_att = TargetAwareAttention(C, num_queries=100, target_size=target_size)
    out = target_att(x)
    assert out.shape == x.shape
    print(f"   Input: {x.shape} -> Output: {out.shape} ✅")
    
    print("\n" + "="*50)
    print("✅ All attention modules (with target_size) passed!")
    print("="*50)