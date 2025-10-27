import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class RetinaHead(nn.Module):
    """RetinaNet Detection Head with proper initialization (添加 target_size 支持)"""
    
    # 1. 在 __init__ 方法参数列表中新增 target_size，放在**kwargs 前
    def __init__(self, in_channels, num_anchors, num_classes=1, target_size=None, **kwargs):
        super().__init__()
        
        if kwargs:
            print(f"🔍 RetinaHead ignored kwargs: {list(kwargs.keys())}")
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # 2. 保存 target_size 为实例属性（供后续尺寸相关计算使用）
        self.target_size = target_size  # 格式为 (H, W)，如 (512, 512)
        # 若未传入 target_size，默认用 512x512（避免后续计算报错）
        if self.target_size is None:
            self.target_size = (512, 512)
            print(f"⚠️ RetinaHead: target_size not provided, using default (512, 512)")
        
        # Classification subnet (4层卷积)
        cls_layers = []
        for _ in range(4):
            cls_layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            ])
        self.cls_subnet = nn.Sequential(*cls_layers)
        self.cls_score = nn.Conv2d(
            in_channels, num_anchors * num_classes, 3, padding=1, bias=True
        )
        
        # Regression subnet (4层卷积)
        reg_layers = []
        for _ in range(4):
            reg_layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            ])
        self.reg_subnet = nn.Sequential(*reg_layers)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, 3, padding=1, bias=True
        )
        
        # 🔥 关键修复2: 初始化权重
        self._init_weights()
        
        # 3. 打印日志时增加 target_size 信息，方便调试
        print(f"✅ RetinaHead initialized: anchors={num_anchors}, classes={num_classes}, target_size={self.target_size}")
    
    def _init_weights(self):
        """Initialize weights for numerical stability"""
        # 分类子网络初始化
        for module in self.cls_subnet.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)
        
        # 🔥 关键修复3: 分类输出层特殊初始化
        # 使用prior probability初始化偏置，这是防止NaN的关键！
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, bias_value)
        
        # 回归子网络初始化
        for module in self.reg_subnet.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)
        
        nn.init.normal_(self.bbox_pred.weight, std=0.01)
        nn.init.constant_(self.bbox_pred.bias, 0)
    
    def forward(self, feats):
        """Forward pass (可基于 self.target_size 扩展尺寸相关逻辑)"""
        cls_scores = []
        bbox_preds = []
        
        for feat in feats:
            # 分类分支
            cls_feat = self.cls_subnet(feat)
            cls_score = self.cls_score(cls_feat)
            cls_scores.append(cls_score)
            
            # 回归分支
            reg_feat = self.reg_subnet(feat)
            bbox_pred = self.bbox_pred(reg_feat)
            bbox_preds.append(bbox_pred)
        
        # （可选）若后续需要将预测框坐标映射回原图尺寸，可基于 self.target_size 计算
        # 示例：bbox_preds = self._map_bbox_to_original_size(bbox_preds)
        
        return cls_scores, bbox_preds