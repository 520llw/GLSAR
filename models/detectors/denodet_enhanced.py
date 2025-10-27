"""
Enhanced DenoDet with Frequency and Attention Modules
集成了频域处理和注意力机制的增强版SAR舰船检测模型（修复tuple赋值问题）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.sar_ship_config import cfg
from models.backbones.fft_resnet import FFTResNet
from models.necks.frequency_fpn import FrequencySpatialFPN
from models.heads.retina_head import RetinaHead
from models.utils.anchor_generator import AnchorGenerator
from models.losses.ciou_loss import CIoULoss

# 导入新模块
from models.utils.frequency import (
    SARFrequencyAttention, 
    FrequencyChannelAttention,
    SpectralGating
)
from models.utils.attention import (
    CBAM, 
    SARScatteringAttention,
    SpeckleNoiseAttention
)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred = torch.clamp(pred, min=-100, max=100)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class EnhancedDenoDet(nn.Module):
    """
    增强版SAR舰船检测模型
    集成频域处理和注意力机制
    """
    
    def __init__(self, use_frequency=True, use_attention=True, target_size=(512, 512)):
        super().__init__()
        
        self.use_frequency = use_frequency
        self.use_attention = use_attention
        self.target_size = target_size
        
        # 过滤配置
        backbone_kwargs = {k: v for k, v in cfg.BACKBONE.items() if k != 'type'}
        neck_kwargs = {k: v for k, v in cfg.NECK.items() if k != 'type'}
        head_kwargs = {k: v for k, v in cfg.HEAD.items() if k != 'type'}
        anchor_kwargs = {k: v for k, v in cfg.ANCHOR_GENERATOR.items() if k != 'type'}
        
        # 基础组件
        self.backbone = FFTResNet(**backbone_kwargs)
        self.neck = FrequencySpatialFPN(**neck_kwargs, target_size=self.target_size)
        self.head = RetinaHead(**head_kwargs, target_size=self.target_size)
        self.anchor_generator = AnchorGenerator(**anchor_kwargs)
        
        # 损失函数
        self.loss_cls = FocalLoss(alpha=cfg.LOSS['focal_loss']['alpha'], 
                                   gamma=cfg.LOSS['focal_loss']['gamma'])
        self.loss_bbox = CIoULoss()
        
        # 🔥 频域增强模块
        if self.use_frequency:
            self.freq_enhance_c3 = SARFrequencyAttention(512, reduction=16, target_size=self.target_size)
            self.freq_enhance_c4 = SARFrequencyAttention(1024, reduction=16, target_size=self.target_size)
            self.freq_enhance_c5 = SARFrequencyAttention(2048, reduction=16, target_size=self.target_size)
            
            self.freq_channel_att = FrequencyChannelAttention(256, reduction=16, target_size=self.target_size)
            self.spectral_gating = SpectralGating(256, gate_channels=16, target_size=self.target_size)
            
            print(f"✅ Frequency modules initialized (target_size: {self.target_size})")
        
        # 🔥 注意力模块
        if self.use_attention:
            self.cbam_c3 = CBAM(512, reduction=16, target_size=self.target_size)
            self.cbam_c4 = CBAM(1024, reduction=16, target_size=self.target_size)
            self.cbam_c5 = CBAM(2048, reduction=16, target_size=self.target_size)
            
            self.sar_scatter_att = SARScatteringAttention(256, target_size=self.target_size)
            self.speckle_att = SpeckleNoiseAttention(256, target_size=self.target_size)
            
            print("✅ Attention modules initialized")
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ Enhanced Model initialized - Total params: {total_params:,}")
    
    def forward(self, inputs, mode='train'):
        """Unified forward with mode switching"""
        if mode == 'train':
            return self.forward_train(inputs)
        else:
            return self.forward_test(inputs)
    
    def forward_train(self, inputs):
        """Training forward pass with enhancements"""
        imgs = inputs['img']
        gt_bboxes = inputs['gt_bboxes']
        gt_labels = inputs['gt_labels']
        
        # ========== Backbone特征提取 ==========
        feats = self.backbone(imgs)  # tuple: (C3, C4, C5)
        
        # 🔥 关键修复：将tuple转换为list以支持修改
        feats = list(feats)
        
        # 🔥 应用频域和注意力增强
        if self.use_frequency:
            feats[0] = self.freq_enhance_c3(feats[0])
            feats[1] = self.freq_enhance_c4(feats[1])
            feats[2] = self.freq_enhance_c5(feats[2])
        
        if self.use_attention:
            feats[0] = self.cbam_c3(feats[0])
            feats[1] = self.cbam_c4(feats[1])
            feats[2] = self.cbam_c5(feats[2])
        
        # ========== Neck特征融合 ==========
        fpn_feats = self.neck(feats)  # tuple: (P3, P4, P5)
        
        # 🔥 FPN特征增强
        enhanced_feats = []
        for feat in fpn_feats:
            enhanced_feat = feat
            
            if self.use_frequency:
                enhanced_feat = self.freq_channel_att(enhanced_feat)
                enhanced_feat = self.spectral_gating(enhanced_feat)
            
            if self.use_attention:
                enhanced_feat = self.sar_scatter_att(enhanced_feat)
                enhanced_feat = self.speckle_att(enhanced_feat)
            
            enhanced_feats.append(enhanced_feat)
        
        # ========== Detection Head ==========
        cls_scores, bbox_preds = self.head(enhanced_feats)
        
        # ========== 生成anchors ==========
        all_anchors = [
            self.anchor_generator.generate_anchors(feat.shape[2:], i, imgs.device)
            for i, feat in enumerate(enhanced_feats)
        ]
        
        # ========== 计算损失 ==========
        cls_loss, bbox_loss = self._compute_loss(
            cls_scores, bbox_preds, all_anchors,
            gt_bboxes, gt_labels, imgs.device,
            imgs.shape[3], imgs.shape[2]
        )
        
        total_loss = cls_loss + bbox_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'bbox_loss': bbox_loss
        }
    
    def forward_test(self, inputs):
        """Inference forward pass"""
        imgs = inputs['img']
        
        # Backbone
        feats = self.backbone(imgs)
        
        # 🔥 关键修复：将tuple转换为list
        feats = list(feats)
        
        # 应用增强
        if self.use_frequency:
            feats[0] = self.freq_enhance_c3(feats[0])
            feats[1] = self.freq_enhance_c4(feats[1])
            feats[2] = self.freq_enhance_c5(feats[2])
        
        if self.use_attention:
            feats[0] = self.cbam_c3(feats[0])
            feats[1] = self.cbam_c4(feats[1])
            feats[2] = self.cbam_c5(feats[2])
        
        # Neck
        fpn_feats = self.neck(feats)
        
        # FPN增强
        enhanced_feats = []
        for feat in fpn_feats:
            enhanced_feat = feat
            if self.use_frequency:
                enhanced_feat = self.freq_channel_att(enhanced_feat)
                enhanced_feat = self.spectral_gating(enhanced_feat)
            if self.use_attention:
                enhanced_feat = self.sar_scatter_att(enhanced_feat)
                enhanced_feat = self.speckle_att(enhanced_feat)
            enhanced_feats.append(enhanced_feat)
        
        # Head
        cls_scores, bbox_preds = self.head(enhanced_feats)
        
        return self._post_process(cls_scores, bbox_preds, imgs.shape[2:])
    
    def _compute_loss(self, cls_scores, bbox_preds, anchors, gt_bboxes, gt_labels, device, W, H):
        """计算损失"""
        # Flatten predictions
        cls_flat = torch.cat([
            s.permute(0, 2, 3, 1).reshape(s.shape[0], -1, cfg.NUM_CLASSES)
            for s in cls_scores
        ], dim=1)
        
        bbox_flat = torch.cat([
            b.permute(0, 2, 3, 1).reshape(b.shape[0], -1, 4)
            for b in bbox_preds
        ], dim=1)
        
        anchors_flat = torch.cat(anchors, dim=0).unsqueeze(0).repeat(cls_flat.shape[0], 1, 1)
        bbox_flat = torch.clamp(bbox_flat, min=-1000, max=1000)
        
        B = cls_flat.shape[0]
        cls_loss_sum = torch.tensor(0.0, device=device)
        bbox_loss_sum = torch.tensor(0.0, device=device)
        num_pos_samples = 0
        
        for b in range(B):
            gt_box = gt_bboxes[b].to(device)
            gt_lab = gt_labels[b].to(device)
            num_gt = len(gt_box)
            
            if num_gt == 0:
                # 多类别：创建全零目标
                cls_tgt = torch.zeros(cls_flat.shape[1], cfg.NUM_CLASSES, device=device)
                cls_loss_sum = cls_loss_sum + self.loss_cls(cls_flat[b], cls_tgt)
                continue
            
            # Convert YOLO to absolute coordinates
            gt_box_abs = gt_box.clone()
            gt_box_abs[:, 0] = (gt_box[:, 0] - gt_box[:, 2] / 2) * W
            gt_box_abs[:, 1] = (gt_box[:, 1] - gt_box[:, 3] / 2) * H
            gt_box_abs[:, 2] = (gt_box[:, 0] + gt_box[:, 2] / 2) * W
            gt_box_abs[:, 3] = (gt_box[:, 1] + gt_box[:, 3] / 2) * H
            gt_box_abs = gt_box_abs.clamp(0, max(W, H))
            
            # 过滤无效boxes
            valid_mask = (gt_box_abs[:, 2] > gt_box_abs[:, 0]) & \
                         (gt_box_abs[:, 3] > gt_box_abs[:, 1])
            if valid_mask.sum() == 0:
                cls_tgt = torch.zeros(cls_flat.shape[1], cfg.NUM_CLASSES, device=device)
                cls_loss_sum = cls_loss_sum + self.loss_cls(cls_flat[b], cls_tgt)
                continue
            
            gt_box_abs = gt_box_abs[valid_mask]
            gt_lab = gt_lab[valid_mask]
            
            # Match anchors
            ious = self._bbox_iou(anchors_flat[b], gt_box_abs)
            max_ious, max_ids = ious.max(dim=1)
            
            # 使用配置的IoU阈值
            pos_iou_thr = cfg.LOSS.get('pos_iou_thr', 0.5)
            fg_mask = max_ious > pos_iou_thr
            
            # 多类别标签
            cls_tgt = torch.zeros(cls_flat.shape[1], cfg.NUM_CLASSES, device=device)
            
            if fg_mask.sum() > 0:
                # 为正样本设置对应类别
                matched_labels = gt_lab[max_ids[fg_mask]]
                cls_tgt[fg_mask, matched_labels] = 1.0
                num_pos_samples += fg_mask.sum().item()
                
                # Bbox loss
                fg_preds = bbox_flat[b][fg_mask]
                fg_gt = gt_box_abs[max_ids[fg_mask]]
                fg_preds = torch.clamp(fg_preds, min=0, max=max(W, H))
                
                bbox_loss = self.loss_bbox(fg_preds, fg_gt)
                bbox_loss_sum = bbox_loss_sum + bbox_loss
            
            cls_loss_sum = cls_loss_sum + self.loss_cls(cls_flat[b], cls_tgt)
        
        cls_loss = cls_loss_sum / B
        
        if num_pos_samples > 0:
            bbox_loss = bbox_loss_sum / num_pos_samples
        else:
            bbox_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # NaN检查
        if torch.isnan(cls_loss):
            print("⚠️ Warning: cls_loss is NaN")
            cls_loss = torch.tensor(0.01, device=device, requires_grad=True)
        
        if torch.isnan(bbox_loss):
            print("⚠️ Warning: bbox_loss is NaN")
            bbox_loss = torch.tensor(0.01, device=device, requires_grad=True)
        
        return cls_loss, bbox_loss
    
    def _bbox_iou(self, boxes1, boxes2):
        """Calculate IoU"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        return inter / (union + 1e-7)
    
    def _post_process(self, cls_scores, bbox_preds, img_shape):
        """Post-process predictions with NMS (多类别版本)"""
        results = []
        B = cls_scores[0].shape[0]
        
        for b in range(B):
            all_scores = []
            all_labels = []
            all_bboxes = []
            
            for level_idx in range(len(cls_scores)):
                cls_score = cls_scores[level_idx][b]
                bbox_pred = bbox_preds[level_idx][b]
                
                num_anchors = 9
                num_classes = cfg.NUM_CLASSES
                H, W = cls_score.shape[1], cls_score.shape[2]
                
                # Reshape: [num_anchors * num_classes, H, W] -> [H, W, num_anchors, num_classes]
                cls_score = cls_score.reshape(num_anchors, num_classes, H, W)
                cls_score = cls_score.permute(2, 3, 0, 1).reshape(-1, num_classes)
                
                bbox_pred = bbox_pred.reshape(num_anchors, 4, H, W)
                bbox_pred = bbox_pred.permute(2, 3, 0, 1).reshape(-1, 4)
                
                # 对每个类别取最大分数
                scores = torch.sigmoid(cls_score)  # [N, num_classes]
                max_scores, labels = scores.max(dim=1)  # [N]
                
                all_scores.append(max_scores)
                all_labels.append(labels)
                all_bboxes.append(bbox_pred)
            
            all_scores = torch.cat(all_scores, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_bboxes = torch.cat(all_bboxes, dim=0)
            
            # 置信度过滤
            keep = all_scores > cfg.CONF_THRESHOLD
            filtered_scores = all_scores[keep]
            filtered_labels = all_labels[keep]
            filtered_bboxes = all_bboxes[keep]
            
            # Per-class NMS
            if len(filtered_scores) > 0 and cfg.NMS_THRESHOLD < 1.0:
                keep_nms = []
                for class_id in range(cfg.NUM_CLASSES):
                    class_mask = filtered_labels == class_id
                    if class_mask.sum() > 0:
                        class_boxes = filtered_bboxes[class_mask]
                        class_scores = filtered_scores[class_mask]
                        class_keep = nms(class_boxes, class_scores, cfg.NMS_THRESHOLD)
                        keep_nms.append(torch.where(class_mask)[0][class_keep])
                
                if keep_nms:
                    keep_nms = torch.cat(keep_nms)
                    filtered_scores = filtered_scores[keep_nms]
                    filtered_labels = filtered_labels[keep_nms]
                    filtered_bboxes = filtered_bboxes[keep_nms]
            
            # Top-K
            if len(filtered_scores) > cfg.MAX_DETECTIONS:
                top_k = torch.topk(filtered_scores, cfg.MAX_DETECTIONS)[1]
                filtered_scores = filtered_scores[top_k]
                filtered_labels = filtered_labels[top_k]
                filtered_bboxes = filtered_bboxes[top_k]
            
            results.append({
                'scores': filtered_scores,
                'labels': filtered_labels,
                'bboxes': filtered_bboxes
            })
        
        return results


# 向后兼容
DenoDet = lambda: EnhancedDenoDet(target_size=(512, 512))


def create_denodet(enhanced=True, use_frequency=True, use_attention=True, target_size=(512, 512)):
    """
    创建DenoDet模型
    
    Args:
        enhanced: 是否使用增强版
        use_frequency: 是否使用频域模块
        use_attention: 是否使用注意力模块
        target_size: 目标尺寸
    
    Returns:
        模型实例
    """
    return EnhancedDenoDet(
        use_frequency=use_frequency,
        use_attention=use_attention,
        target_size=target_size
    )