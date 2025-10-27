import cv2
import numpy as np
import torch
import random
from config import DATA_CONFIG


class Compose:
    """组合多个变换"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class Resize:
    """调整图像和边界框大小"""
    def __init__(self, size=None):
        self.size = self._get_valid_size(size or DATA_CONFIG['img_size'])
        
    def _get_valid_size(self, size):
        if isinstance(size, int):
            return (size, size)
        elif isinstance(size, (tuple, list)) and len(size) == 2:
            return (size[0], size[1])
        else:
            raise ValueError(f"无效的尺寸格式: {size}，请使用整数或 (h, w) 元组")
        
    def __call__(self, data):
        img = data['img']
        bboxes = data['gt_bboxes']
        
        h, w = img.shape[:2]
        new_h, new_w = self.size
        
        # 调整图像大小
        img = cv2.resize(img, (new_w, new_h))
        
        # 调整边界框大小
        scale_h = new_h / h
        scale_w = new_w / w
        
        if len(bboxes) > 0:
            bboxes[:, 0] *= scale_w  # x1
            bboxes[:, 1] *= scale_h  # y1
            bboxes[:, 2] *= scale_w  # x2
            bboxes[:, 3] *= scale_h  # y2
        
        data['img'] = img
        data['gt_bboxes'] = bboxes
        data['img_shape'] = (new_h, new_w)
        return data


class RandomFlip:
    """随机翻转图像和边界框"""
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, data):
        img = data['img']
        bboxes = data['gt_bboxes']
        h, w = img.shape[:2]
        
        if random.random() < self.prob:
            if random.random() < 0.5:
                # 水平翻转
                img = cv2.flip(img, 1)
                if len(bboxes) > 0:
                    x1 = w - bboxes[:, 2]
                    x2 = w - bboxes[:, 0]
                    bboxes[:, 0] = x1
                    bboxes[:, 2] = x2
            else:
                # 垂直翻转
                img = cv2.flip(img, 0)
                if len(bboxes) > 0:
                    y1 = h - bboxes[:, 3]
                    y2 = h - bboxes[:, 1]
                    bboxes[:, 1] = y1
                    bboxes[:, 3] = y2
        
        data['img'] = img
        data['gt_bboxes'] = bboxes
        return data


class Normalize:
    """图像归一化，支持单通道"""
    def __init__(self, mean=None, std=None, force_single_channel=False):
        self.mean = np.array(mean or [0.485], dtype=np.float32)
        self.std = np.array(std or [0.229], dtype=np.float32)
        self.force_single_channel = force_single_channel
        
    def __call__(self, data):
        img = data['img'].astype(np.float32)
        
        # 确保单通道并添加通道维度
        if self.force_single_channel:
            if img.ndim == 3 and img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.ndim == 2:  # 确保有通道维度
                img = np.expand_dims(img, axis=-1)
        
        # 归一化
        img = img / 255.0
        img = (img - self.mean) / self.std
        
        data['img'] = img
        return data


class ToTensor:
    """将数据转换为Tensor，修复轴变换错误"""
    def __init__(self, force_single_channel=False):
        self.force_single_channel = force_single_channel
        
    def __call__(self, data):
        img = data['img']
        
        # 关键修复：确保图像有通道维度
        if img.ndim == 2:
            # 增加通道维度 (H, W) -> (H, W, 1)
            img = np.expand_dims(img, axis=-1)
        elif img.ndim == 3 and img.shape[-1] not in (1, 3):
            # 处理异常维度
            raise ValueError(f"不支持的图像维度: {img.shape}，预期为 (H, W) 或 (H, W, 1/3)")
        
        # 转换为Tensor并调整通道顺序 (HWC -> CHW)
        # 现在img一定是3维的 (H, W, C)，可以安全地转置
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        # 边界框转换为Tensor
        bboxes = torch.from_numpy(data['gt_bboxes']).float() if len(data['gt_bboxes']) > 0 else torch.empty((0, 4))
        labels = torch.from_numpy(data['gt_labels']).long() if len(data['gt_labels']) > 0 else torch.empty(0, dtype=torch.long)
        
        return {
            'img': img,
            'gt_bboxes': bboxes,
            'gt_labels': labels,
            'img_path': data['img_path'],
            'img_shape': data['img_shape']
        }


class RandomCrop:
    """随机裁剪图像和边界框"""
    def __init__(self, size=None, padding=0):
        self.size = self._get_valid_size(size or DATA_CONFIG['img_size'])
        self.padding = padding
        
    def _get_valid_size(self, size):
        if isinstance(size, int):
            return (size, size)
        elif isinstance(size, (tuple, list)) and len(size) == 2:
            return (size[0], size[1])
        else:
            raise ValueError(f"无效的尺寸格式: {size}，请使用整数或 (h, w) 元组")
        
    def __call__(self, data):
        img = data['img']
        bboxes = data['gt_bboxes'].copy()
        
        # 填充
        if self.padding > 0:
            border_mode = cv2.BORDER_CONSTANT
            if img.ndim == 3:
                value = [0, 0, 0] if img.shape[-1] == 3 else [0]
            else:
                value = 0
                
            img = cv2.copyMakeBorder(
                img, self.padding, self.padding, self.padding, self.padding,
                border_mode, value=value
            )
            if len(bboxes) > 0:
                bboxes += self.padding
        
        h, w = img.shape[:2]
        new_h, new_w = self.size
        
        if w == new_w and h == new_h:
            return data
            
        x1 = random.randint(0, w - new_w)
        y1 = random.randint(0, h - new_h)
        x2 = x1 + new_w
        y2 = y1 + new_h
        
        # 裁剪图像
        img = img[y1:y2, x1:x2]
        
        # 调整边界框
        if len(bboxes) > 0:
            bboxes[:, 0] -= x1
            bboxes[:, 1] -= y1
            bboxes[:, 2] -= x1
            bboxes[:, 3] -= y1
            
            mask = (
                (bboxes[:, 0] < new_w) &
                (bboxes[:, 1] < new_h) &
                (bboxes[:, 2] > 0) &
                (bboxes[:, 3] > 0)
            )
            bboxes = bboxes[mask]
            
            bboxes[:, 0] = np.clip(bboxes[:, 0], 0, new_w)
            bboxes[:, 1] = np.clip(bboxes[:, 1], 0, new_h)
            bboxes[:, 2] = np.clip(bboxes[:, 2], 0, new_w)
            bboxes[:, 3] = np.clip(bboxes[:, 3], 0, new_h)
        
        data['img'] = img
        data['gt_bboxes'] = bboxes
        data['img_shape'] = (new_h, new_w)
        return data


class ForceSingleChannel:
    """强制将图像转换为单通道的变换"""
    def __call__(self, data):
        img = data['img']
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 确保有通道维度
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        data['img'] = img
        return data


# 常用变换组合，支持force_single_channel参数
def train_transforms(force_single_channel=False):
    transforms_list = [
        Resize(),
        RandomFlip(prob=0.5),
        RandomCrop()
    ]
    
    if force_single_channel:
        transforms_list.insert(0, ForceSingleChannel())
    
    transforms_list.extend([
        Normalize(force_single_channel=force_single_channel),
        ToTensor(force_single_channel=force_single_channel)
    ])
    
    return Compose(transforms_list)


def test_transforms(force_single_channel=False):
    transforms_list = [Resize()]
    
    if force_single_channel:
        transforms_list.insert(0, ForceSingleChannel())
    
    transforms_list.extend([
        Normalize(force_single_channel=force_single_channel),
        ToTensor(force_single_channel=force_single_channel)
    ])
    
    return Compose(transforms_list)
