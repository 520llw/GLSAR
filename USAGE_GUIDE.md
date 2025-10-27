### 4. 运行完整性检查
```bash
python check_modules.py
```

这会检查：
- ✅ PyTorch安装
- ✅ 所有依赖包
- ✅ 配置文件
- ✅ 模型文件
- ✅ 数据文件
- ✅ 训练脚本

### 5. 准备数据
```bash
# 将你的数据放到对应目录
# 图像格式: .jpg, .png
# 标签格式: YOLO格式 (.txt)

# 检查数据
python -c "
from pathlib import Path
train_imgs = list(Path('data/train/images').glob('*'))
train_labels = list(Path('data/train/labels').glob('*'))
print(f'Train images: {len(train_imgs)}')
print(f'Train labels: {len(train_labels)}')
"
```

### 6. 测试模型
```bash
# 快速测试
python test_model.py

# 如果成功，你会看到：
# ✅ All tests passed!
```

### 7. 开始训练
```bash
# 完整增强版本
python train_enhanced.py

# 或选择特定配置
python train_enhanced.py --no_frequency  # 仅attention
python train_enhanced.py --baseline      # 基线版本
```

---

## 🔧 故障排除速查表

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| ModuleNotFoundError: models.utils.attention | 文件缺失 | 创建 `models/utils/attention.py` |
| CUDA out of memory | 显存不足 | `--batch_size 4` 或 `--no_frequency` |
| Loss is NaN | 学习率过大 | 降低学习率到 `5e-5` |
| No improvement | 数据问题 | 检查标签格式和数据质量 |
| Import error | 路径问题 | 确保在项目根目录运行 |
| Gradient explosion | 梯度裁剪不够 | 设置 `GRAD_CLIP_MAX_NORM = 5.0` |

---

## 📚 代码组织最佳实践

### 1. 模块化设计
```python
# 好的做法 ✅
from models.utils.attention import CBAM
from models.utils.frequency import SARFrequencyAttention

class MyModel(nn.Module):
    def __init__(self):
        self.cbam = CBAM(256)
        self.freq_att = SARFrequencyAttention(256)

# 不好的做法 ❌
# 把所有代码写在一个文件里
```

### 2. 配置管理
```python
# 好的做法 ✅
from config.sar_ship_config import cfg

model = EnhancedDenoDet().to(cfg.DEVICE)
optimizer = cfg.get_optimizer(model)

# 不好的做法 ❌
# 硬编码参数
model = EnhancedDenoDet().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
```

### 3. 实验管理
```bash
# 好的做法 ✅
# 使用有意义的实验名称
python train_enhanced.py --exp_name resnet50_cbam_freq_lr1e4

# 不好的做法 ❌
python train_enhanced.py  # 使用随机生成的名称
```

---

## 🎯 快速开始清单

### ☑️ 第一次使用

- [ ] 1. 运行 `python check_modules.py` 检查环境
- [ ] 2. 创建 `models/utils/attention.py` 和 `frequency.py`
- [ ] 3. 确保数据目录结构正确
- [ ] 4. 运行 `python test_model.py --test data` 测试数据加载
- [ ] 5. 运行 `python test_model.py` 完整测试
- [ ] 6. 开始训练 `python train_enhanced.py`

### ☑️ 每次训练前

- [ ] 1. 检查数据路径在 `config/sar_ship_config.py` 中正确
- [ ] 2. 确认 batch size 和学习率合适
- [ ] 3. 选择合适的增强模块配置
- [ ] 4. 设置有意义的实验名称
- [ ] 5. 确保有足够的磁盘空间保存checkpoint

### ☑️ 训练过程中

- [ ] 1. 监控训练曲线（每5个epoch查看一次）
- [ ] 2. 检查loss是否正常下降
- [ ] 3. 观察学习率调度是否合理
- [ ] 4. 注意是否有OOM错误
- [ ] 5. 定期查看验证集性能

### ☑️ 训练完成后

- [ ] 1. 运行 `python tools/visualize.py` 查看预测
- [ ] 2. 分析训练曲线找出问题
- [ ] 3. 对比不同配置的性能
- [ ] 4. 保存最佳模型和配置
- [ ] 5. 记录实验结果

---

## 💻 命令速查表

```bash
# ============ 测试相关 ============
# 完整测试
python test_model.py

# 单项测试
python test_model.py --test data      # 测试数据
python test_model.py --test forward   # 测试前向传播
python test_model.py --test modules   # 测试模块
python test_model.py --test optimizer # 测试优化器

# 检查环境
python check_modules.py


# ============ 训练相关 ============
# 完整增强训练
python train_enhanced.py

# Baseline训练
python train_enhanced.py --baseline

# 部分增强
python train_enhanced.py --no_frequency
python train_enhanced.py --no_attention
python train_enhanced.py --no_diff_lr

# 自定义参数
python train_enhanced.py \
  --epochs 100 \
  --batch_size 16 \
  --exp_name my_exp

# 断点续训
python train_enhanced.py --resume latest
python train_enhanced.py --resume checkpoints/checkpoint_epoch_50.pth


# ============ 可视化相关 ============
# 基础可视化
python tools/visualize.py --checkpoint checkpoints/best_model.pth

# 完整可视化
python tools/visualize.py \
  --checkpoint checkpoints/best_model.pth \
  --num_samples 50 \
  --conf_threshold 0.5 \
  --show_gt \
  --save_crops


# ============ 调试相关 ============
# 打印模型结构
python -c "
from models.detectors.denodet import EnhancedDenoDet
model = EnhancedDenoDet()
print(model)
"

# 统计参数量
python -c "
from models.detectors.denodet import EnhancedDenoDet
model = EnhancedDenoDet()
total = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total:,}')
"

# 检查数据
python -c "
from data.loaders import get_train_dataloader
loader = get_train_dataloader()
batch = next(iter(loader))
print('Batch keys:', batch.keys())
print('Image shape:', batch['img'].shape)
print('GT boxes:', [len(b) for b in batch['gt_bboxes']])
"
```

---

## 🎓 进阶技巧

### 1. 学习率查找器
```python
# 创建 tools/find_lr.py
import torch
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt

def find_lr(model, train_loader, init_lr=1e-8, final_lr=10, beta=0.98):
    """Learning rate range test"""
    num_batches = len(train_loader)
    mult = (final_lr / init_lr) ** (1/num_batches)
    lr = init_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    avg_loss = 0
    best_loss = 0
    losses = []
    lrs = []
    
    for batch_idx, data in enumerate(train_loader):
        # Forward
        loss = model(data)['total_loss']
        
        # Smooth loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**(batch_idx+1))
        
        # Record
        losses.append(smoothed_loss)
        lrs.append(lr)
        
        # Stop if loss explodes
        if batch_idx > 0 and smoothed_loss > 4 * best_loss:
            break
        
        if smoothed_loss < best_loss or batch_idx == 0:
            best_loss = smoothed_loss
        
        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update LR
        lr *= mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True)
    plt.savefig('lr_finder.png')
    print(f"Suggested LR: {lrs[losses.index(min(losses))]:.2e}")
```

### 2. 梯### 2. 打印模型结构
```python
# 在训练前添加
from torchsummary import summary
summary(model, input_size=(1, 512, 512))
```

### 3. 监控梯度
```python
# 在 engine/trainer.py 的 train_epoch 中添加
for name, param in self.model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 100:
            print(f"Large gradient: {name} = {grad_norm:.2f}")
```

### 4. 可视化特征图
```python
# 创建 tools/visualize_features.py
import torch
import matplotlib.pyplot as plt

def visualize_feature_maps(model, img, layer_name):
    """可视化中间层特征图"""
    features = {}
    
    def hook(module, input, output):
        features['output'] = output
    
    # 注册hook
    handle = getattr(model, layer_name).register_forward_hook(hook)
    
    # Forward
    with torch.no_grad():
        model(img)
    
    # 可视化
    feat = features['output'][0]  # [C, H, W]
    plt.figure(figsize=(20, 4))
    for i in range(min(8, feat.shape[0])):
        plt.subplot(1, 8, i+1)
        plt.imshow(feat[i].cpu(), cmap='jet')
        plt.axis('off')
    plt.show()
    
    handle.remove()
```

---

## 📦 完整安装指南

### 1. 克隆/设置项目
```bash
# 创建项目目录
mkdir sar_ship_detection
cd sar_ship_detection

# 创建必要的目录结构
mkdir -p config models/{detectors,backbones,necks,heads,utils,losses}
mkdir -p data/{train,val}/{images,labels}
mkdir -p engine tools checkpoints logs
```

### 2. 安装依赖
```bash
# 创建虚拟环境（推荐）
conda create -n sar_detection python=3.8
conda activate sar_detection

# 安装PyTorch（根据CUDA版本选择）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 或 CPU only
pip install torch torchvision

# 安装其他依赖
pip install opencv-python numpy matplotlib tqdm
```

### 3. 复制代码文件
将以下文件复制到对应位置：
- ✅ `config/sar_ship_config.py`
- ✅ `models/utils/attention.py`
- ✅ `models/utils/frequency.py`
- ✅ `train_enhanced.py`
- ✅ `test_model.py`
- ✅ `tools/visualize.py`
- ✅ `check_modules.py`

### 4. 运行完整性检查
```bash# SAR Ship Detection - 使用指南

## 📋 目录
1. [环境配置](#环境配置)
2. [测试模型](#测试模型)
3. [训练模型](#训练模型)
4. [可视化预测](#可视化预测)
5. [常见问题](#常见问题)

---

## 🔧 环境配置

### 必要依赖
```bash
pip install torch torchvision
pip install opencv-python numpy matplotlib
pip install tqdm pathlib
```

### 目录结构
```
project/
├── config/
│   └── sar_ship_config.py       # ✅ 已修复
├── models/
│   ├── detectors/
│   │   └── denodet.py           # 增强版检测器
│   ├── backbones/
│   ├── necks/
│   ├── heads/
│   └── utils/
│       ├── attention.py
│       └── frequency.py
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
├── engine/
│   └── trainer.py
├── tools/
│   └── visualize.py             # ✅ 已修复
├── train_enhanced.py            # ✅ 已修复
├── test_model.py                # ✅ 已修复
└── checkpoints/
```

---

## 🧪 测试模型

### 1. 运行完整测试
```bash
python test_model.py
```

这将测试：
- ✅ 数据加载
- ✅ 模型前向传播
- ✅ 反向传播
- ✅ 推理模式
- ✅ 增强模块
- ✅ 优化器配置

### 2. 运行单项测试
```bash
# 仅测试数据加载
python test_model.py --test data

# 仅测试模型前向传播
python test_model.py --test forward

# 仅测试增强模块
python test_model.py --test modules

# 仅测试优化器
python test_model.py --test optimizer
```

### 预期输出示例
```
================================================================================
🚀 SAR Ship Detection - Model Testing Suite
================================================================================
Device: cuda
Batch Size: 8
Image Size: (512, 512)
================================================================================

🧪 Testing Data Loading
✅ Batch loaded successfully
✅ Channel check passed
✅ Batch size check passed
✅ Image size check passed

🧪 Testing Model Forward Pass
✅ Model initialized with 43,234,567 parameters
✅ Forward pass successful
✅ Loss validity check passed
✅ Backward pass successful
✅ Gradient check passed
✅ Inference successful

📊 Test Summary
Total tests: 4
✅ Passed: 4
❌ Failed: 0
🎉 All tests passed successfully!
```

---

## 🚀 训练模型

### 1. 基础训练（完整增强）
```bash
python train_enhanced.py
```

默认配置：
- ✅ Attention模块启用
- ✅ Frequency模块启用
- ✅ 差异化学习率启用

### 2. Baseline训练（无增强）
```bash
python train_enhanced.py --baseline
```

### 3. 部分增强训练

#### 仅使用Attention模块
```bash
python train_enhanced.py --no_frequency
```

#### 仅使用Frequency模块
```bash
python train_enhanced.py --no_attention
```

#### 使用增强但不用差异化学习率
```bash
python train_enhanced.py --no_diff_lr
```

### 4. 自定义训练参数
```bash
python train_enhanced.py \
  --epochs 50 \
  --batch_size 16 \
  --exp_name my_experiment \
  --no_val  # 禁用验证
```

### 5. 断点续训
```bash
# 从最新checkpoint继续
python train_enhanced.py --resume latest

# 从指定checkpoint继续
python train_enhanced.py --resume checkpoints/checkpoint_epoch_20.pth
```

### 训练输出
```
================================================================================
SAR Ship Detection Training - Enhanced Version
================================================================================
📝 Experiment: sar_att_freq_difflr_20241011_143022
🖥️  Device: cuda
🔄 Epochs: 100
📦 Batch Size: 8
🖼️  Image Size: (512, 512)
🌱 Seed: 42
✨ Attention Modules: ✅
📡 Frequency Modules: ✅
🎓 Differential LR: ✅
================================================================================

🔧 Initializing model...
✅ Frequency modules initialized
✅ Attention modules initialized
✅ Enhanced Model initialized - Total params: 43,234,567
✅ Total parameters: 43,234,567
✅ Trainable parameters: 43,234,567

📦 Loading datasets...
✅ Train batches: 1250
✅ Val batches: 156

🎯 Creating trainer...
🔥 Using differential learning rates for enhanced modules

📊 Parameter Groups:
  Group 0: 21,234,567 params, lr=1.00e-05  # Backbone
  Group 1: 5,123,456 params, lr=1.00e-04   # Attention
  Group 2: 4,876,544 params, lr=1.00e-04   # Frequency
  Group 3: 8,000,000 params, lr=1.00e-04   # Neck
  Group 4: 4,000,000 params, lr=1.00e-04   # Head
```

### 训练监控

#### 实时日志
```
Epoch 1/100 - train_loss: 2.3456 (cls: 1.2345, bbox: 1.1111), lr: 1.00e-04
Epoch 2/100 - train_loss: 2.1234 (cls: 1.1234, bbox: 1.0000), val_loss: 2.2345, lr: 9.80e-05
...
```

#### 训练曲线
自动保存到：`logs/{experiment_name}/curves/training_curves.png`

包含4个子图：
- 📉 总损失（训练 + 验证）
- 📉 分类损失
- 📉 边界框损失
- 📉 学习率变化

#### Checkpoint保存
- `checkpoints/best_model.pth` - 最佳模型
- `checkpoints/latest.pth` - 最新模型
- `checkpoints/checkpoint_epoch_N.pth` - 每5个epoch保存

---

## 📊 可视化预测

### 1. 基础可视化
```bash
python tools/visualize.py --checkpoint checkpoints/best_model.pth
```

### 2. 自定义参数
```bash
python tools/visualize.py \
  --checkpoint checkpoints/best_model.pth \
  --num_samples 50 \
  --conf_threshold 0.5 \
  --show_gt \
  --save_crops \
  --output_dir my_visualizations
```

### 参数说明
- `--checkpoint`: 模型权重路径（必需）
- `--num_samples`: 可视化样本数量（默认10）
- `--conf_threshold`: 置信度阈值（默认0.3）
- `--show_gt`: 显示ground truth（绿色框）
- `--save_crops`: 保存检测目标的裁剪图
- `--output_dir`: 输出目录（默认visualizations）
- `--data_dir`: 自定义数据目录

### 输出示例
```
================================================================================
SAR Ship Detection - Prediction Visualization
================================================================================
Checkpoint: checkpoints/best_model.pth
Output directory: visualizations
Confidence threshold: 0.3
Show ground truth: True
================================================================================

📥 Loading model from checkpoints/best_model.pth
   Epoch: 45
   Best loss: 0.5432
✅ Model loaded successfully

📦 Loading dataset...
✅ Loaded 1000 images

📊 Processing 50 samples...
Visualizing: 100%|████████████████████| 50/50 [00:23<00:00,  2.11it/s]

================================================================================
📊 Statistics
================================================================================
Images processed: 50
Total detections: 127
Avg detections/image: 2.54
High confidence (>0.5): 98
Total ground truth: 132
Avg GT/image: 2.64
================================================================================

✅ Visualizations saved to: visualizations
✅ Detection crops saved to: visualizations/crops/
```

### 可视化说明
- 🟢 **绿色框**: Ground truth（真实标注）
- 🔴 **红色框**: 模型预测
- 每个框显示：类别 + 置信度分数

---

## ❓ 常见问题

### Q1: 显存不足（CUDA out of memory）
```bash
# 方案1: 减小batch size
python train_enhanced.py --batch_size 4

# 方案2: 减小图像尺寸（需修改config）
# 在 config/sar_ship_config.py 中:
IMG_SIZE = (384, 384)  # 原来是 (512, 512)

# 方案3: 禁用部分增强模块
python train_enhanced.py --no_frequency  # 或 --no_attention
```

### Q2: 损失为NaN
```bash
# 方案1: 降低学习率
# 在 config/sar_ship_config.py 中:
OPTIMIZER = {
    'lr': 5e-5,  # 原来是 1e-4
    ...
}

# 方案2: 启用梯度裁剪（已默认启用）
GRAD_CLIP_MAX_NORM = 5.0  # 原来是 10.0

# 方案3: 检查数据
python test_model.py --test data
```

### Q3: 训练速度慢
```bash
# 方案1: 减少数据加载worker数
# 在 config/sar_ship_config.py 中:
NUM_WORKERS = 2  # 原来是 4

# 方案2: 禁用某些增强模块
python train_enhanced.py --no_attention --no_frequency

# 方案3: 使用baseline模型
python train_enhanced.py --baseline
```

### Q4: 模型不收敛
```bash
# 方案1: 增加warmup epochs
# 在 config/sar_ship_config.py 中:
WARMUP_EPOCHS = 10  # 原来是 5

# 方案2: 调整学习率策略
SCHEDULER = {
    'type': 'StepLR',
    'step_size': 20,
    'gamma': 0.5
}

# 方案3: 禁用差异化学习率
python train_enhanced.py --no_diff_lr
```

### Q5: 如何对比不同配置？
```bash
# 训练baseline
python train_enhanced.py --baseline --exp_name baseline

# 训练仅用attention
python train_enhanced.py --no_frequency --exp_name only_attention

# 训练仅用frequency
python train_enhanced.py --no_attention --exp_name only_frequency

# 训练完整版本
python train_enhanced.py --exp_name full_enhanced

# 查看所有实验结果
ls logs/  # 每个实验有独立目录
```

---

## 📈 性能优化建议

### 1. 数据增强
如果训练集较小，可以在 `data/datasets.py` 中启用更多增强：
- 随机旋转
- 随机翻转
- 随机亮度/对比度调整
- Mixup / CutMix

### 2. 学习率调优
```python
# 找到最佳学习率 - 在 config/sar_ship_config.py
OPTIMIZER = {
    'type': 'AdamW',
    'lr': 1e-4,  # 可尝试: 5e-5, 1e-4, 5e-4
    'weight_decay': 1e-4,
}

# 差异化学习率调优
DIFFERENTIAL_LR = {
    'enabled': True,
    'backbone_lr_mult': 0.1,     # Backbone用较小学习率
    'attention_lr_mult': 1.0,    # 新模块用标准学习率
    'frequency_lr_mult': 1.0,
    'neck_lr_mult': 1.0,
    'head_lr_mult': 1.0,
}
```

### 3. 损失函数权重调整
```python
# 在 config/sar_ship_config.py
LOSS = {
    'focal_loss': {
        'alpha': 0.25,      # 正负样本平衡
        'gamma': 2.0,       # 难易样本权重
        'weight': 1.0
    },
    'ciou_loss': {
        'weight': 1.0       # 可调整为 0.5-2.0
    }
}
```

### 4. Anchor配置优化
根据你的数据集中目标的尺寸分布调整：
```python
ANCHOR_GENERATOR = {
    'scales': [8, 16, 32],      # 根据目标大小调整
    'ratios': [0.5, 1.0, 2.0],  # 根据目标长宽比调整
    'strides': [8, 16, 32]
}
```

---

## 🔬 高级用法

### 1. 自定义模型配置

创建自己的配置类：
```python
# my_config.py
from config.sar_ship_config import SARShipConfig

class MyConfig(SARShipConfig):
    # 覆盖需要修改的参数
    BATCH_SIZE = 16
    EPOCHS = 150
    IMG_SIZE = (640, 640)
    
    ATTENTION_CONFIG = {
        'use_attention': True,
        'cbam_reduction': 8,  # 更强的注意力
        ...
    }

cfg = MyConfig()
```

### 2. 模块消融实验

创建实验脚本：
```bash
#!/bin/bash
# ablation_study.sh

# Baseline
python train_enhanced.py --baseline --exp_name exp_baseline --epochs 50

# Only Attention
python train_enhanced.py --no_frequency --exp_name exp_attention --epochs 50

# Only Frequency
python train_enhanced.py --no_attention --exp_name exp_frequency --epochs 50

# Attention + Frequency (No Diff LR)
python train_enhanced.py --no_diff_lr --exp_name exp_both_nodiff --epochs 50

# Full (Attention + Frequency + Diff LR)
python train_enhanced.py --exp_name exp_full --epochs 50
```

运行：
```bash
chmod +x ablation_study.sh
./ablation_study.sh
```

### 3. 多GPU训练

修改 `train_enhanced.py`:
```python
# 在main()函数中添加
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
```

### 4. 混合精度训练监控

查看AMP是否正常工作：
```python
# 在 engine/trainer.py 的 train_epoch 中添加
if self.use_amp and batch_idx % 100 == 0:
    print(f"Scaler scale: {self.scaler.get_scale():.2f}")
```

---

## 📝 实验记录模板

### 实验配置表格
| 实验名称 | Attention | Frequency | Diff LR | Batch Size | Epochs | Best mAP |
|---------|-----------|-----------|---------|------------|--------|----------|
| baseline | ❌ | ❌ | ❌ | 8 | 50 | 0.XXX |
| att_only | ✅ | ❌ | ❌ | 8 | 50 | 0.XXX |
| freq_only | ❌ | ✅ | ❌ | 8 | 50 | 0.XXX |
| full | ✅ | ✅ | ✅ | 8 | 50 | 0.XXX |

### 训练日志分析

检查训练曲线：
```bash
# 查看所有实验的曲线
ls logs/*/curves/training_curves.png

# 使用图像查看器对比
eog logs/*/curves/training_curves.png  # Linux
open logs/*/curves/training_curves.png  # Mac
```

---

## 🐛 调试技巧

### 1. 启用异常检测
```python
# 在 config/sar_ship_config.py
DEBUG = {
    'detect_anomaly': True,  # 检测NaN/Inf
    'print_model': True      # 打印模型结构
}
```

### 2. 打印模型结构
```python
# 在训练前添加
from