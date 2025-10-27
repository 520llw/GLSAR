"""
Data Loaders for SAR Ship Detection
修复版：正确引用配置参数
"""

import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets import SARShipDataset
from config.sar_ship_config import cfg


def collate_fn(batch):
    """
    Custom collate function for batching
    处理不同数量的ground truth boxes
    """
    imgs = []
    bboxes = []
    labels = []
    img_paths = []
    
    for sample in batch:
        imgs.append(sample['img'])
        bboxes.append(sample['gt_bboxes'])
        labels.append(sample['gt_labels'])
        img_paths.append(sample['img_path'])
    
    # Stack images: [B, C, H, W]
    imgs = torch.stack(imgs, dim=0)
    
    return {
        'img': imgs,
        'gt_bboxes': bboxes,  # List of tensors
        'gt_labels': labels,  # List of tensors
        'img_paths': img_paths
    }


def get_train_dataloader():
    """创建训练数据加载器"""
    dataset = SARShipDataset(
        img_dir=cfg.TRAIN_IMG_DIR,
        label_dir=cfg.TRAIN_LABEL_DIR,
        img_size=cfg.IMG_SIZE,
        mode='train'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=getattr(cfg, 'PIN_MEMORY', True),  # 安全获取，默认True
        drop_last=True  # 丢弃最后不完整的batch
    )
    
    return dataloader


def get_val_dataloader():
    """创建验证数据加载器"""
    dataset = SARShipDataset(
        img_dir=cfg.VAL_IMG_DIR,
        label_dir=cfg.VAL_LABEL_DIR,
        img_size=cfg.IMG_SIZE,
        mode='val'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=getattr(cfg, 'PIN_MEMORY', True),
        drop_last=False
    )
    
    return dataloader


def get_test_dataloader(test_img_dir=None, test_label_dir=None):
    """创建测试数据加载器"""
    if test_img_dir is None:
        test_img_dir = cfg.VAL_IMG_DIR  # 默认使用验证集
    if test_label_dir is None:
        test_label_dir = cfg.VAL_LABEL_DIR
    
    dataset = SARShipDataset(
        img_dir=test_img_dir,
        label_dir=test_label_dir,
        img_size=cfg.IMG_SIZE,
        mode='test'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 测试时batch_size=1
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=getattr(cfg, 'PIN_MEMORY', True),
        drop_last=False
    )
    
    return dataloader


# 向后兼容的函数名
def build_dataloader(split='train'):
    """构建数据加载器（向后兼容）"""
    if split == 'train':
        return get_train_dataloader()
    elif split == 'val':
        return get_val_dataloader()
    elif split == 'test':
        return get_test_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'")


if __name__ == '__main__':
    """测试数据加载器"""
    print("="*80)
    print("Testing Data Loaders")
    print("="*80)
    
    # 测试训练加载器
    print("\n1️⃣ Testing train dataloader...")
    train_loader = get_train_dataloader()
    print(f"   ✅ Train loader created: {len(train_loader)} batches")
    
    # 加载一个batch
    batch = next(iter(train_loader))
    print(f"   📦 Batch info:")
    print(f"      Images shape: {batch['img'].shape}")
    print(f"      Num samples: {len(batch['gt_bboxes'])}")
    print(f"      Boxes per sample: {[len(b) for b in batch['gt_bboxes'][:3]]}")
    print(f"      Labels per sample: {[len(l) for l in batch['gt_labels'][:3]]}")
    
    # 检查类别范围
    all_labels = torch.cat(batch['gt_labels'])
    if len(all_labels) > 0:
        print(f"      Label range: {all_labels.min().item()} - {all_labels.max().item()}")
        print(f"      Unique labels: {all_labels.unique().tolist()}")
    
    # 测试验证加载器
    print("\n2️⃣ Testing val dataloader...")
    val_loader = get_val_dataloader()
    print(f"   ✅ Val loader created: {len(val_loader)} batches")
    
    print("\n" + "="*80)
    print("✅ All dataloaders tested successfully!")
    print("="*80)