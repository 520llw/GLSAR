import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

# Suppress albumentations warnings
logging.getLogger("albumentations").setLevel(logging.WARNING)

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.sar_ship_config import cfg


class SARShipDataset(Dataset):
    """SAR Ship Dataset - Enforced Single Channel"""
    
    def __init__(self, img_dir, label_dir, img_size=(800, 800), mode='train'):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.mode = mode
        self.img_files = sorted(list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png')))
        
        if len(self.img_files) == 0:
            raise ValueError(f"No images found in {img_dir}")
        
        # Augmentation
        p_aug = 0.5 if mode == 'train' else 0.0
        self.transform = A.Compose([
            A.Resize(height=img_size[1], width=img_size[0]),
            A.HorizontalFlip(p=p_aug),
            A.VerticalFlip(p=p_aug * 0.6),
            A.RandomRotate90(p=p_aug * 0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=p_aug),
            A.Normalize(mean=[0.5], std=[0.5]),
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
        
        # Force grayscale loading
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load: {img_path}")
        
        # Ensure 3D shape (H, W, 1)
        img = np.expand_dims(img, axis=2).astype(np.uint8)
        
        # Load YOLO labels
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
        
        # Apply transforms
        transformed = self.transform(image=img, bboxes=bboxes_np, labels=labels_np)
        img_transformed = transformed['image']
        
        # CRITICAL: Force single channel if transform produced 3 channels
        if img_transformed.shape[0] == 3:
            img_transformed = img_transformed.mean(dim=0, keepdim=True)
        
        # Ensure single channel
        if img_transformed.shape[0] != 1:
            raise ValueError(f"Expected 1 channel, got {img_transformed.shape[0]}")
        
        bboxes_transformed = (torch.tensor(transformed['bboxes']) 
                            if len(transformed['bboxes']) > 0 
                            else torch.zeros((0, 4)))
        labels_transformed = (torch.tensor(transformed['labels'], dtype=torch.long) 
                            if len(transformed['labels']) > 0 
                            else torch.zeros(0, dtype=torch.long))
        
        return {
            'img': img_transformed,
            'gt_bboxes': bboxes_transformed,
            'gt_labels': labels_transformed,
            'img_path': str(img_path)
        }


def collate_fn(batch):
    """Custom collate with final channel safety check"""
    imgs = torch.stack([item['img'] for item in batch])
    
    # Final safety check
    if imgs.shape[1] != 1:
        imgs = imgs.mean(dim=1, keepdim=True)
        print(f"⚠️ Batch collate forced to 1 channel")
    
    return {
        'img': imgs,
        'gt_bboxes': [item['gt_bboxes'] for item in batch],
        'gt_labels': [item['gt_labels'] for item in batch],
        'img_path': [item['img_path'] for item in batch]
    }