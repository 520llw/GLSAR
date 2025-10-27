import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AnchorGenerator:
    """Generate anchors for detection"""
    
    def __init__(self, base_sizes, ratios, scales, strides, **kwargs):
        self.base_sizes = base_sizes
        self.ratios = ratios
        self.scales = scales
        self.strides = strides
        
        if kwargs:
            print(f"🔍 AnchorGenerator ignored kwargs: {list(kwargs.keys())}")
    
    def generate_anchors(self, feat_shape, level, device):
        """Generate anchors for a feature level"""
        H, W = feat_shape
        stride = self.strides[level]
        base_size = self.base_sizes[level]
        
        # Generate base anchors
        anchors = []
        for ratio in self.ratios:
            for scale in self.scales:
                w = base_size * scale * torch.sqrt(torch.tensor(ratio))
                h = base_size * scale / torch.sqrt(torch.tensor(ratio))
                anchors.append([-w/2, -h/2, w/2, h/2])
        
        base_anchors = torch.tensor(anchors, device=device, dtype=torch.float32)
        
        # Generate grid
        shift_x = torch.arange(0, W, device=device, dtype=torch.float32) * stride
        shift_y = torch.arange(0, H, device=device, dtype=torch.float32) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=2).reshape(-1, 4)
        
        # Generate all anchors
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.reshape(-1, 4)
        
        return all_anchors


# ============================================================================
# models/losses/ciou_loss.py
# ============================================================================
import torch
import torch.nn as nn


class CIoULoss(nn.Module):
    """Complete IoU Loss for bbox regression"""
    
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] (x1, y1, x2, y2)
            target_boxes: [N, 4] (x1, y1, x2, y2)
        """
        # IoU
        iou = self._calculate_iou(pred_boxes, target_boxes)
        
        # Center distance
        pred_center = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        target_center = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
        center_dist = torch.sum((pred_center - target_center) ** 2, dim=1)
        
        # Diagonal of enclosing box
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
        
        # Aspect ratio
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]
        
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_w / (target_h + self.eps)) - 
            torch.atan(pred_w / (pred_h + self.eps)), 2
        )
        alpha = v / (1 - iou + v + self.eps)
        
        ciou = iou - (center_dist / (enclose_diagonal + self.eps)) - alpha * v
        loss = 1 - ciou
        
        return loss.mean()
    
    def _calculate_iou(self, boxes1, boxes2):
        """Calculate IoU"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        union_area = area1 + area2 - inter_area
        iou = inter_area / (union_area + self.eps)
        
        return iou
