import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config.sar_ship_config import cfg
import logging
import torch  # 加torch for squeeze
logging.getLogger("albumentations").setLevel(logging.WARNING)

class SARShipDataset(Dataset):
    """SAR Ship Dataset (YOLO + 终极1C灰度强制)"""
    
    def __init__(self, img_dir=cfg.TRAIN_IMG_DIR, label_dir=cfg.TRAIN_LABEL_DIR, img_size=(800, 800), mode='train'):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.mode = mode
        self.img_files = sorted(self.img_dir.glob("*.jpg"))
        if len(self.img_files) == 0:
            raise ValueError(f"No images in {img_dir}")
        
        # Compose: 1C + SAR aug (val弱化)
        p_aug = 0.5 if mode == 'train' else 0.0
        self.transform = A.Compose([
            A.Resize(height=img_size[1], width=img_size[0]),
            A.HorizontalFlip(p=p_aug),
            A.VerticalFlip(p=p_aug * 0.6),
            A.RandomRotate90(p=p_aug * 0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=p_aug),
            A.Normalize(mean=[0.5], std=[0.5]),  # 单通道
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['labels'],
            min_visibility=0.3,
            min_area=0.01
        ))
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # 1C (H,W)
        if img is None:
            raise ValueError(f"Load fail: {img_path}")
        
        # 强3D: (H,W,1)
        img = np.expand_dims(img, axis=2).astype(np.uint8)
        
        # YOLO labels (原有)
        bboxes_list, labels_list = [], []
        label_path = self.label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        bboxes_list.append([xc, yc, bw, bh])
                        labels_list.append(cls_id)
        
        bboxes_np = np.array(bboxes_list) if bboxes_list else np.empty((0, 4))
        labels_np = np.array(labels_list) if labels_list else np.empty(0)
        
        # Transform
        transformed = self.transform(image=img, bboxes=bboxes_np, labels=labels_np)
        img_transformed = transformed['image']  # 可能[3,H,W]
        
        # **终极修复**：强制1C (mean squeeze，SAR灰度无损)
        if img_transformed.shape[0] == 3:
            print(f"⚠️ Force squeeze 3C→1C for {img_path.stem}")  # 调试，后删
            img_transformed = img_transformed.mean(dim=0, keepdim=True)  # [1,H,W] 均值灰度
        
        bboxes_transformed = torch.tensor(transformed['bboxes']) if len(transformed['bboxes']) > 0 else torch.zeros((0, 4))
        labels_transformed = torch.tensor(transformed['labels']) if len(transformed['labels']) > 0 else torch.zeros(0, dtype=torch.long)
        
        # 调试print (首批跑加)
        if idx < 2:  # 只前2张
            print(f"🚀 Sample {idx}: img.shape={img_transformed.shape}, bboxes.shape={bboxes_transformed.shape}")
        
        return {
            'img': img_transformed,  # 铁1C
            'gt_bboxes': bboxes_transformed,
            'gt_labels': labels_transformed,
            'img_path': str(img_path)
        }
