import torch
from torch.utils.data import DataLoader
# 移除COCODataset，导入修改后的YOLOSARDataset和get_dataset
from .datasets import YOLOSARDataset, get_dataset  
from .transforms import train_transforms, test_transforms
from config import DATA_CONFIG


def collate_fn(batch):
    """批处理函数，增加通道检查"""
    imgs = []
    bboxes = []
    labels = []
    img_paths = []
    img_shapes = []
    
    for data in batch:
        img = data['img']
        # 关键检查1：确保图像是单通道
        if img.ndim == 3 and img.shape[0] == 3:
            # 若为3通道，强制转为单通道（取平均值）
            img = torch.mean(img, dim=0, keepdim=True)
        elif img.ndim == 2:
            # 若为2D张量（H,W），添加通道维度
            img = img.unsqueeze(0)
            
        imgs.append(img)
        bboxes.append(data['gt_bboxes'])
        labels.append(data['gt_labels'])
        img_paths.append(data['img_path'])
        img_shapes.append(data['img_shape'])
    
    # 堆叠图像张量 (B, C, H, W)，此时C应为1
    imgs = torch.stack(imgs, dim=0)
    
    # 调试：打印批次的通道数和尺寸
    # print(f"批次图像尺寸: {imgs.shape} (B, C, H, W)")
    
    return {
        'img': imgs,
        'gt_bboxes': bboxes,
        'gt_labels': labels,
        'img_paths': img_paths,
        'img_shapes': img_shapes
    }


def build_dataloader(split='train', dataset_type='sar'):
    """构建数据加载器，确保加载单通道图像"""
    if dataset_type == 'sar':
        # 获取数据集时强制单通道处理
        dataset = get_dataset(
            split=split,
            # 关键修改：在transform中明确要求单通道
            transform=train_transforms(force_single_channel=True) 
                      if split == 'train' else 
                      test_transforms(force_single_channel=True)
        )
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}（仅支持'sar'）")
    
    dataloader = DataLoader(
        dataset,
        batch_size=DATA_CONFIG['batch_size'],
        shuffle=(split == 'train'),
        num_workers=DATA_CONFIG['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


def get_train_dataloader(dataset_type='sar'):
    return build_dataloader(split='train', dataset_type=dataset_type)


def get_val_dataloader(dataset_type='sar'):
    return build_dataloader(split='val', dataset_type=dataset_type)


def get_test_dataloader(dataset_type='sar'):
    return build_dataloader(split='test', dataset_type=dataset_type)
