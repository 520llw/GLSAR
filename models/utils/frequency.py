"""
Frequency Domain Modules for SAR Ship Detection
频域处理模块，专为SAR图像设计（修复 target_size 支持）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SARFrequencyAttention(nn.Module):
    """
    SAR Frequency Attention
    在频域进行特征增强，利用SAR图像的频谱特性
    """
    
    def __init__(self, channels, reduction=16, target_size=None):
        """
        Args:
            channels: 输入通道数
            reduction: 通道压缩比例
            target_size: 目标尺寸 (H, W)，用于尺寸适配
        """
        super().__init__()
        self.channels = channels
        self.target_size = target_size if target_size is not None else (512, 512)
        
        # 频域特征处理
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 空域特征处理
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        print(f"✅ SARFrequencyAttention initialized: channels={channels}, target_size={self.target_size}")
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        identity = x
        
        # 适配目标尺寸（如果需要）
        if (H, W) != self.target_size:
            x_resized = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        else:
            x_resized = x
        
        # ========== FFT变换到频域 ==========
        fft = torch.fft.rfft2(x_resized, norm='ortho')
        fft_mag = torch.abs(fft)  # 幅度谱
        fft_phase = torch.angle(fft)  # 相位谱
        
        # ========== 频域注意力 ==========
        # 对幅度谱进行注意力加权
        att = self.freq_conv(fft_mag)
        fft_mag_att = fft_mag * att
        
        # ========== 重构复数频谱 ==========
        fft_real = fft_mag_att * torch.cos(fft_phase)
        fft_imag = fft_mag_att * torch.sin(fft_phase)
        fft_enhanced = torch.complex(fft_real, fft_imag)
        
        # ========== IFFT变换回空域 ==========
        freq_out = torch.fft.irfft2(fft_enhanced, s=self.target_size, norm='ortho')
        
        # 恢复原始尺寸
        if freq_out.shape[2:] != identity.shape[2:]:
            freq_out = F.interpolate(freq_out, size=identity.shape[2:], mode='bilinear', align_corners=False)
        
        # ========== 空域增强 ==========
        spatial_out = self.spatial_conv(identity)
        
        # ========== 融合 ==========
        out = freq_out + spatial_out
        
        return out


class FrequencyChannelAttention(nn.Module):
    """
    Frequency-based Channel Attention
    基于频域信息的通道注意力
    """
    
    def __init__(self, channels, reduction=16, target_size=None):
        super().__init__()
        self.channels = channels
        self.target_size = target_size if target_size is not None else (512, 512)
        
        # 通道注意力网络
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
        # 频域池化权重
        self.freq_weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        
        print(f"✅ FrequencyChannelAttention initialized: channels={channels}, target_size={self.target_size}")
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 适配目标尺寸
        if (H, W) != self.target_size:
            x_resized = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        else:
            x_resized = x
        
        # ========== 频域全局池化 ==========
        # FFT
        fft = torch.fft.rfft2(x_resized, norm='ortho')
        fft_mag = torch.abs(fft)
        
        # 加权池化频域特征
        freq_pool = torch.mean(fft_mag * self.freq_weight, dim=[2, 3])  # [B, C]
        
        # ========== 通道注意力 ==========
        channel_att = self.fc(freq_pool).view(B, C, 1, 1)  # [B, C, 1, 1]
        
        # ========== 应用注意力 ==========
        out = x * channel_att
        
        return out


class SpectralGating(nn.Module):
    """
    Spectral Gating Mechanism
    频谱门控，用于过滤频域噪声
    """
    
    def __init__(self, channels, gate_channels=16, target_size=None):
        super().__init__()
        self.channels = channels
        self.target_size = target_size if target_size is not None else (512, 512)
        
        # 门控网络
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, gate_channels, 1),
            nn.BatchNorm2d(gate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels, channels, 1),
            nn.Sigmoid()
        )
        
        # 频域滤波器学习
        self.freq_filter = nn.Parameter(torch.ones(1, channels, 1, 1))
        
        print(f"✅ SpectralGating initialized: channels={channels}, target_size={self.target_size}")
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # ========== 生成门控权重 ==========
        gate = self.gate_conv(x)
        
        # 适配目标尺寸
        if (H, W) != self.target_size:
            x_resized = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        else:
            x_resized = x
        
        # ========== FFT ==========
        fft = torch.fft.rfft2(x_resized, norm='ortho')
        fft_mag = torch.abs(fft)
        fft_phase = torch.angle(fft)
        
        # ========== 频域滤波 ==========
        # 应用学习的频域滤波器
        fft_mag_filtered = fft_mag * self.freq_filter
        
        # 重构
        fft_real = fft_mag_filtered * torch.cos(fft_phase)
        fft_imag = fft_mag_filtered * torch.sin(fft_phase)
        fft_filtered = torch.complex(fft_real, fft_imag)
        
        # ========== IFFT ==========
        freq_out = torch.fft.irfft2(fft_filtered, s=self.target_size, norm='ortho')
        
        # 恢复原始尺寸
        if freq_out.shape[2:] != x.shape[2:]:
            freq_out = F.interpolate(freq_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # ========== 门控调制 ==========
        out = x + freq_out * gate
        
        return out


class GaborFilter(nn.Module):
    """
    Gabor Filter Bank
    Gabor滤波器组，用于多方向特征提取
    """
    
    def __init__(self, channels, num_orientations=8, num_scales=3):
        super().__init__()
        self.channels = channels
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * num_orientations * num_scales, 
                     channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def _generate_gabor_kernel(self, size, wavelength, orientation, 
                               sigma=None, gamma=0.5):
        """生成Gabor核"""
        if sigma is None:
            sigma = 0.56 * wavelength
        
        # 创建网格
        x = torch.arange(-size // 2, size // 2 + 1, dtype=torch.float32)
        y = torch.arange(-size // 2, size // 2 + 1, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        # 旋转
        x_theta = x * math.cos(orientation) + y * math.sin(orientation)
        y_theta = -x * math.sin(orientation) + y * math.cos(orientation)
        
        # Gabor函数
        gaussian = torch.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * math.pi * x_theta / wavelength)
        
        gabor = gaussian * sinusoid
        gabor = gabor / gabor.sum()
        
        return gabor
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Gabor参数
        kernel_size = 7
        base_wavelength = 4.0
        
        # 存储所有方向和尺度的响应
        responses = []
        
        for scale_idx in range(self.num_scales):
            wavelength = base_wavelength * (2 ** scale_idx)
            
            for orient_idx in range(self.num_orientations):
                orientation = orient_idx * math.pi / self.num_orientations
                
                # 生成Gabor核
                gabor_kernel = self._generate_gabor_kernel(
                    kernel_size, wavelength, orientation
                ).to(device)
                
                # 扩展为[C_out, C_in, H, W]格式
                gabor_kernel = gabor_kernel.unsqueeze(0).unsqueeze(0)
                gabor_kernel = gabor_kernel.repeat(C, 1, 1, 1)
                
                # 应用Gabor滤波
                response = F.conv2d(x, gabor_kernel, 
                                   padding=kernel_size // 2, 
                                   groups=C)
                responses.append(response)
        
        # 拼接所有响应
        all_responses = torch.cat(responses, dim=1)
        
        # 融合
        out = self.fusion(all_responses)
        
        return out


class WaveletTransform(nn.Module):
    """
    Discrete Wavelet Transform (DWT)
    小波变换，用于多尺度特征分解
    """
    
    def __init__(self, channels, num_levels=3):
        super().__init__()
        self.channels = channels
        self.num_levels = num_levels
        
        # 每个level的处理网络
        self.level_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 4, channels, 1),  # 4个子带
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_levels)
        ])
        
        # 重建网络
        self.recon_conv = nn.Sequential(
            nn.Conv2d(channels * num_levels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def _dwt2d(self, x):
        """2D离散小波变换（简化版本，使用Haar小波）"""
        B, C, H, W = x.shape
        
        # 平均池化（近似系数 LL）
        ll = F.avg_pool2d(x, 2)
        
        # 高频系数
        # LH (水平)
        lh = F.avg_pool2d(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
        lh = F.pad(lh, (0, 0, 0, 0), mode='replicate')
        
        # HL (垂直)
        hl = F.avg_pool2d(x[:, :, 1:, :] - x[:, :, :-1, :], 2)
        hl = F.pad(hl, (0, 0, 0, 0), mode='replicate')
        
        # HH (对角)
        hh = F.avg_pool2d(
            (x[:, :, 1:, 1:] - x[:, :, :-1, :-1]), 2
        )
        hh = F.pad(hh, (0, 0, 0, 0), mode='replicate')
        
        return ll, lh, hl, hh
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        level_features = []
        current = x
        
        # 多级小波分解
        for level_idx in range(self.num_levels):
            ll, lh, hl, hh = self._dwt2d(current)
            
            # 拼接4个子带
            subbands = torch.cat([ll, lh, hl, hh], dim=1)
            
            # 处理
            level_feat = self.level_convs[level_idx](subbands)
            
            # 上采样到原始尺寸
            if level_feat.shape[2:] != x.shape[2:]:
                level_feat = F.interpolate(level_feat, size=x.shape[2:],
                                          mode='bilinear', align_corners=False)
            
            level_features.append(level_feat)
            
            # 下一级使用LL子带
            current = ll
        
        # 融合所有level
        all_features = torch.cat(level_features, dim=1)
        out = self.recon_conv(all_features)
        
        return out


class FourierUnit(nn.Module):
    """
    Fourier Unit (FNO-style)
    傅里叶神经算子风格的频域处理单元
    """
    
    def __init__(self, channels, modes=32):
        super().__init__()
        self.channels = channels
        self.modes = modes  # 保留的傅里叶模式数
        
        # 频域权重（复数）
        scale = 1 / (channels * channels)
        self.weights_real = nn.Parameter(
            scale * torch.randn(channels, channels, modes, modes // 2 + 1)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.randn(channels, channels, modes, modes // 2 + 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # ========== FFT ==========
        x_fft = torch.fft.rfft2(x)
        
        # ========== 频域操作 ==========
        out_fft = torch.zeros_like(x_fft)
        
        # 只处理低频模式
        h_modes = min(self.modes, H)
        w_modes = min(self.modes, W // 2 + 1)
        
        # 复数乘法
        weights = torch.complex(self.weights_real, self.weights_imag)
        
        # [B, C_out, C_in, h, w] @ [B, C_in, h, w] -> [B, C_out, h, w]
        x_fft_slice = x_fft[:, :, :h_modes, :w_modes]
        
        # Einstein summation for complex multiplication
        out_fft[:, :, :h_modes, :w_modes] = torch.einsum(
            'oihw,bihw->bohw',
            weights[:, :, :h_modes, :w_modes],
            x_fft_slice
        )
        
        # ========== IFFT ==========
        out = torch.fft.irfft2(out_fft, s=(H, W))
        
        return out


# ========== 测试代码 ==========
if __name__ == '__main__':
    print("Testing Frequency Modules (with target_size support)...")
    
    # 测试参数
    B, C, H, W = 2, 256, 64, 64
    target_size = (128, 128)
    x = torch.randn(B, C, H, W)
    
    # 测试SARFrequencyAttention
    print("\n1. Testing SARFrequencyAttention...")
    freq_att = SARFrequencyAttention(C, reduction=16, target_size=target_size)
    out = freq_att(x)
    assert out.shape == x.shape
    print(f"   Input: {x.shape} -> Output: {out.shape} ✅")
    
    # 测试FrequencyChannelAttention
    print("\n2. Testing FrequencyChannelAttention...")
    freq_ch_att = FrequencyChannelAttention(C, reduction=16, target_size=target_size)
    out = freq_ch_att(x)
    assert out.shape == x.shape
    print(f"   Input: {x.shape} -> Output: {out.shape} ✅")
    
    # 测试SpectralGating
    print("\n3. Testing SpectralGating...")
    spectral_gate = SpectralGating(C, gate_channels=16, target_size=target_size)
    out = spectral_gate(x)
    assert out.shape == x.shape
    print(f"   Input: {x.shape} -> Output: {out.shape} ✅")
    
    # 测试GaborFilter
    print("\n4. Testing GaborFilter...")
    gabor = GaborFilter(C, num_orientations=8, num_scales=3)
    out = gabor(x)
    assert out.shape == x.shape
    print(f"   Input: {x.shape} -> Output: {out.shape} ✅")
    
    # 测试WaveletTransform
    print("\n5. Testing WaveletTransform...")
    wavelet = WaveletTransform(C, num_levels=3)
    out = wavelet(x)
    assert out.shape == x.shape
    print(f"   Input: {x.shape} -> Output: {out.shape} ✅")
    
    # 测试FourierUnit
    print("\n6. Testing FourierUnit...")
    fourier = FourierUnit(C, modes=32)
    out = fourier(x)
    assert out.shape == x.shape
    print(f"   Input: {x.shape} -> Output: {out.shape} ✅")
    
    print("\n" + "="*50)
    print("✅ All frequency modules passed!")
    print("="*50)