"""
数据集类别分布检查脚本
检查是否存在类别不平衡问题
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm

from config.sar_ship_config import cfg
from data.loaders import get_train_dataloader


def analyze_dataset_distribution():
    """详细分析数据集分布"""
    print("="*80)
    print("📊 SAR舰船数据集分布分析")
    print("="*80)
    
    # 1. 统计标签文件
    print("\n1️⃣ 统计标签文件...")
    label_files = list(cfg.TRAIN_LABEL_DIR.glob('*.txt'))
    
    class_counter = Counter()
    samples_per_class = defaultdict(list)
    empty_files = 0
    total_objects = 0
    
    for label_file in tqdm(label_files, desc="扫描标签"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            empty_files += 1
            continue
        
        file_classes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                class_counter[class_id] += 1
                file_classes.append(class_id)
                total_objects += 1
        
        for cls in set(file_classes):
            samples_per_class[cls].append(label_file.stem)
    
    # 打印统计结果
    print(f"\n📋 标签文件统计:")
    print(f"   总标签文件: {len(label_files)}")
    print(f"   空标签文件: {empty_files}")
    print(f"   有效文件: {len(label_files) - empty_files}")
    print(f"   总目标数: {total_objects}")
    
    print(f"\n📊 类别分布:")
    print(f"{'类别ID':<8} {'类别名称':<12} {'目标数':<10} {'样本数':<10} {'占比':<10}")
    print("-"*60)
    
    for class_id in range(cfg.NUM_CLASSES):
        class_name = cfg.CLASSES[class_id]
        count = class_counter.get(class_id, 0)
        num_samples = len(samples_per_class.get(class_id, []))
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        
        print(f"{class_id:<8} {class_name:<12} {count:<10} {num_samples:<10} {percentage:<10.2f}%")
    
    # 2. 检查不平衡程度
    print(f"\n⚠️  数据不平衡分析:")
    counts = [class_counter.get(i, 0) for i in range(cfg.NUM_CLASSES)]
    if max(counts) > 0 and min(counts) >= 0:
        imbalance_ratio = max(counts) / max(min(counts), 1)
        print(f"   最大类别/最小类别比例: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            print(f"   🚨 严重不平衡！(>{10}:1)")
        elif imbalance_ratio > 5:
            print(f"   ⚠️  中度不平衡 (>{5}:1)")
        elif imbalance_ratio > 2:
            print(f"   ⚠️  轻度不平衡 (>{2}:1)")
        else:
            print(f"   ✅ 分布较均衡")
    
    # 3. 测试数据加载器的批次分布
    print(f"\n2️⃣ 测试数据加载器批次分布...")
    
    loader = get_train_dataloader()
    
    batch_class_dist = []
    num_batches_to_check = min(100, len(loader))
    
    for i, batch in enumerate(tqdm(loader, total=num_batches_to_check, desc="检查batches")):
        if i >= num_batches_to_check:
            break
        
        all_labels = torch.cat(batch['gt_labels'])
        if len(all_labels) > 0:
            batch_classes = all_labels.unique().tolist()
            batch_class_dist.append(len(batch_classes))
    
    print(f"\n📦 前{num_batches_to_check}个Batch的类别分布:")
    print(f"   平均每batch包含类别数: {np.mean(batch_class_dist):.2f}")
    print(f"   最小类别数/batch: {min(batch_class_dist)}")
    print(f"   最大类别数/batch: {max(batch_class_dist)}")
    
    if np.mean(batch_class_dist) < 2:
        print(f"   🚨 警告：每个batch平均类别数太少！")
        print(f"      这会导致训练不稳定，建议:")
        print(f"      1. 增加batch_size")
        print(f"      2. 使用类别均衡采样")
    
    # 4. 推荐的类别权重
    print(f"\n3️⃣ 推荐的类别权重配置:")
    
    if total_objects > 0:
        # 计算逆频率权重
        weights = {}
        avg_count = total_objects / cfg.NUM_CLASSES
        
        print(f"\n📊 基于逆频率的权重建议:")
        for class_id in range(cfg.NUM_CLASSES):
            class_name = cfg.CLASSES[class_id]
            count = class_counter.get(class_id, 1)
            weight = avg_count / count
            weights[class_name] = weight
            
            print(f"   '{class_name}': {weight:.2f},")
        
        print(f"\n💡 建议在 config/sar_ship_config.py 中更新 CLASS_WEIGHTS:")
        print(f"   CLASS_WEIGHTS = {{")
        for class_name, weight in weights.items():
            print(f"       '{class_name}': {weight:.2f},")
        print(f"   }}")
    
    # 5. 检查是否需要采样策略
    print(f"\n4️⃣ 采样策略建议:")
    
    if imbalance_ratio > 5:
        print(f"   🎯 建议使用类别均衡采样 (Balanced Sampling)")
        print(f"   实现方式:")
        print(f"   1. WeightedRandomSampler")
        print(f"   2. 过采样少数类")
        print(f"   3. 欠采样多数类")
    elif imbalance_ratio > 2:
        print(f"   💡 可以考虑:")
        print(f"   1. 调整类别权重")
        print(f"   2. 使用Focal Loss (已启用)")
    else:
        print(f"   ✅ 当前分布可以直接训练")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == "__main__":
    analyze_dataset_distribution()