import torch
import torch.nn as nn
import math

class CIoULoss(nn.Module):
    """Complete IoU Loss for bbox regression - 添加数值保护"""
    
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] (x1, y1, x2, y2)
            target_boxes: [N, 4] (x1, y1, x2, y2)
        """
        if pred_boxes.shape[0] == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        # 🔥 关键修复8: 数值范围限制
        pred_boxes = torch.clamp(pred_boxes, min=-1e4, max=1e4)
        target_boxes = torch.clamp(target_boxes, min=-1e4, max=1e4)
        
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
        
        # 🔥 关键修复9: 防止除零
        enclose_diagonal = torch.clamp(enclose_diagonal, min=self.eps)
        
        # Aspect ratio
        pred_w = torch.clamp(pred_boxes[:, 2] - pred_boxes[:, 0], min=self.eps)
        pred_h = torch.clamp(pred_boxes[:, 3] - pred_boxes[:, 1], min=self.eps)
        target_w = torch.clamp(target_boxes[:, 2] - target_boxes[:, 0], min=self.eps)
        target_h = torch.clamp(target_boxes[:, 3] - target_boxes[:, 1], min=self.eps)
        
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2
        )
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)
        
        ciou = iou - (center_dist / enclose_diagonal) - alpha * v
        loss = 1 - ciou
        
        # 🔥 关键修复10: 限制最终损失范围
        loss = torch.clamp(loss, min=0, max=2)
        
        return loss.mean()
    
    def _calculate_iou(self, boxes1, boxes2):
        """Calculate IoU"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # 🔥 关键修复11: 确保面积非负
        area1 = torch.clamp(area1, min=self.eps)
        area2 = torch.clamp(area2, min=self.eps)
        
        inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        union_area = area1 + area2 - inter_area
        iou = inter_area / (union_area + self.eps)
        
        return torch.clamp(iou, min=0, max=1)
