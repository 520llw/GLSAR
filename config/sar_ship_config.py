import torch
import torch.optim as optim
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

class SARShipConfig:
    """
    Complete SAR Ship Detection Configuration - 6 Classes Version
    优化版配置，提升训练效果（适配YOLO数据格式）
    """
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========== Data paths（适配YOLO格式：data2/images/train）==========
    DATA_ROOT = PROJECT_ROOT / 'data' / 'data2'
    TRAIN_IMG_DIR = DATA_ROOT / 'images' / 'train'
    TRAIN_LABEL_DIR = DATA_ROOT / 'labels' / 'train'
    VAL_IMG_DIR = DATA_ROOT / 'images' / 'val'
    VAL_LABEL_DIR = DATA_ROOT / 'labels' / 'val'
    
    # Output paths
    CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
    LOG_DIR = PROJECT_ROOT / 'logs'
    
    # ========== 训练参数优化 ==========
    BATCH_SIZE = 4  # 降低batch size，提升稳定性（从8->4）
    EPOCHS = 150    # 增加训练轮数（从120->150）
    IMG_SIZE = (512, 512)
    NUM_WORKERS = 4
    PIN_MEMORY = True  # 加速数据传输到GPU
    SEED = 42
    
    # ========== 6类别配置 ==========
    CLASSES = [
        'cargo',      # 0 - 货船
        'tanker',     # 1 - 油轮
        'container',  # 2 - 集装箱船
        'fishing',    # 3 - 渔船
        'passenger',  # 4 - 客船
        'military'    # 5 - 军舰
    ]
    NUM_CLASSES = 6
    
    # 类别颜色（BGR格式用于OpenCV）
    CLASS_COLORS = {
        'cargo': (0, 255, 0),      # 绿色
        'tanker': (255, 0, 0),     # 蓝色
        'container': (0, 255, 255), # 黄色
        'fishing': (255, 0, 255),   # 紫色
        'passenger': (255, 165, 0), # 橙色
        'military': (128, 0, 128)   # 深紫色
    }
    
    # ========== 类别权重优化（处理数据不平衡）==========
    CLASS_WEIGHTS = {
        'cargo': 1.0,
        'tanker': 1.1,      # 稍微增加权重
        'container': 1.3,   # 提高权重（从1.2->1.3）
        'fishing': 2.0,     # 大幅提高（从1.5->2.0），小目标需要更多关注
        'passenger': 1.2,   # 提高权重（从1.1->1.2）
        'military': 1.5     # 提高权重（从1.3->1.5）
    }
    
    # ========== 优化器参数调整 ==========
    OPTIMIZER = {
        'type': 'AdamW',
        'lr': 3e-5,         # 降低学习率（从5e-5->3e-5），提升稳定性
        'weight_decay': 5e-5, # 增加正则化（从1e-4->5e-5）
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }
    
    # ========== 学习率调度器优化 ==========
    SCHEDULER = {
        'type': 'CosineAnnealingLR',
        'T_max': 150,       # 匹配新的epochs
        'eta_min': 5e-7     # 降低最小学习率（从1e-6->5e-7）
    }
    
    # Model architecture
    BACKBONE = {
        'type': 'FFTResNet',
        'depth': 50,
        'in_channels': 1,
        'num_stages': 4,
        'out_indices': (1, 2, 3),
        'frozen_stages': -1,
        'use_fft': True
    }
    
    NECK = {
        'type': 'FrequencySpatialFPN',
        'in_channels': [512, 1024, 2048],
        'out_channels': 256,
        'num_outs': 3,
        'use_frequency': True
    }
    
    HEAD = {
        'type': 'RetinaHead',
        'num_classes': 6,
        'in_channels': 256,
        'feat_channels': 256,
        'num_anchors': 9
    }
    
    # ========== Anchor Generator 优化 ==========
    ANCHOR_GENERATOR = {
        'type': 'AnchorGenerator',
        'base_sizes': [8, 16, 32],
        'scales': [0.7, 0.9, 1.0, 1.2, 1.5, 2.0],  # 增加scale范围，更好适应多尺度
        'ratios': [0.4, 0.7, 1.0, 1.5, 2.5],        # 增加ratio范围，适应各种船型
        'strides': [8, 16, 32]
    }
    
    # ========== Loss 参数优化 ==========
    LOSS = {
        'focal_loss': {
            'alpha': 0.30,    # 增加alpha（从0.25->0.30），更关注困难样本
            'gamma': 2.5,     # 增加gamma（从2.0->2.5），进一步抑制易分样本
            'weight': 1.0
        },
        'ciou_loss': {
            'eps': 1e-7,
            'weight': 1.2     # 增加bbox loss权重（从1.0->1.2）
        },
        'frequency_loss': {
            'enabled': True,  # 启用频域loss
            'weight': 0.08,   # 降低权重（从0.1->0.08），避免过拟合
        },
        'pos_iou_thr': 0.6,   # 提高正样本IoU阈值（从0.5->0.6），更严格
        'neg_iou_thr': 0.3    # 降低负样本IoU阈值（从0.4->0.3），扩大负样本范围
    }
    
    # ========== Attention和Frequency模块优化 ==========
    ATTENTION_CONFIG = {
        'use_attention': True,
        'attention_types': ['cbam', 'scatter', 'speckle'],  # 减少类型，避免过度复杂
        'cbam_reduction': 16,
        'attention_positions': ['layer2', 'layer3', 'layer4'],
    }
    
    FREQUENCY_CONFIG = {
        'use_frequency': True,
        'frequency_types': ['fft_attention', 'gabor'],  # 减少类型（移除wavelet）
        'gabor_orientations': 6,     # 减少方向数（从8->6）
        'gabor_scales': 2,           # 减少尺度（从3->2）
        'wavelet_levels': 2,         # 减少层级（从3->2）
        'fourier_modes': 24,         # 减少模式数（从32->24）
    }
    
    # ========== 差异化学习率优化 ==========
    DIFFERENTIAL_LR = {
        'enabled': True,
        'backbone_lr_mult': 0.1,      # 保持backbone较低学习率
        'attention_lr_mult': 1.5,     # 提高attention学习率（从1.0->1.5）
        'frequency_lr_mult': 1.5,     # 提高frequency学习率（从1.0->1.5）
        'neck_lr_mult': 1.2,          # 提高neck学习率（从1.0->1.2）
        'head_lr_mult': 2.0,          # 大幅提高head学习率（从1.0->2.0）
    }
    
    # ========== 训练设置优化 ==========
    USE_AMP = True
    GRAD_CLIP_MAX_NORM = 5.0      # 降低梯度裁剪阈值（从10.0->5.0），更稳定
    WARMUP_EPOCHS = 15            # 增加warmup（从10->15）
    EARLY_STOP_PATIENCE = 30      # 增加patience（从25->30）
    
    # Logging
    LOG_INTERVAL = 50
    SAVE_INTERVAL = 5
    VAL_INTERVAL = 1
    
    # ========== 推理参数优化 ==========
    CONF_THRESHOLD = 0.25         # 降低阈值（从0.3->0.25），检测更多目标
    NMS_THRESHOLD = 0.45          # 降低NMS阈值（从0.5->0.45），减少重复框
    MAX_DETECTIONS = 100
    
    # Debug
    DEBUG = {
        'detect_anomaly': False,
        'print_model': False
    }
    
    @classmethod
    def get_optimizer(cls, model):
        """Get standard optimizer"""
        optimizer_type = cls.OPTIMIZER['type']
        optimizer_cls = getattr(optim, optimizer_type)
        
        return optimizer_cls(
            model.parameters(),
            lr=cls.OPTIMIZER['lr'],
            weight_decay=cls.OPTIMIZER['weight_decay'],
            betas=cls.OPTIMIZER.get('betas', (0.9, 0.999)),
            eps=cls.OPTIMIZER.get('eps', 1e-8)
        )
    
    @classmethod
    def get_optimizer_enhanced(cls, model):
        """Get optimizer with differential learning rates (优化版)"""
        if not cls.DIFFERENTIAL_LR['enabled']:
            return cls.get_optimizer(model)
        
        params = []
        param_names = set()
        
        # Backbone参数（最低学习率）
        backbone_params = []
        for n, p in model.named_parameters():
            if 'backbone' in n and not any(x in n for x in ['attention', 'freq', 'gabor', 'wavelet']):
                backbone_params.append(p)
                param_names.add(n)
        if backbone_params:
            params.append({
                'params': backbone_params,
                'lr': cls.OPTIMIZER['lr'] * cls.DIFFERENTIAL_LR['backbone_lr_mult'],
                'name': 'backbone'
            })
        
        # Attention模块参数（高学习率）
        attention_params = []
        for n, p in model.named_parameters():
            if any(x in n for x in ['attention', 'cbam', 'scatter', 'speckle']) and n not in param_names:
                attention_params.append(p)
                param_names.add(n)
        if attention_params:
            params.append({
                'params': attention_params,
                'lr': cls.OPTIMIZER['lr'] * cls.DIFFERENTIAL_LR['attention_lr_mult'],
                'name': 'attention'
            })
        
        # Frequency模块参数（高学习率）
        freq_params = []
        for n, p in model.named_parameters():
            if any(x in n for x in ['freq', 'gabor', 'wavelet', 'fourier']) and n not in param_names:
                freq_params.append(p)
                param_names.add(n)
        if freq_params:
            params.append({
                'params': freq_params,
                'lr': cls.OPTIMIZER['lr'] * cls.DIFFERENTIAL_LR['frequency_lr_mult'],
                'name': 'frequency'
            })
        
        # Neck参数（中等学习率）
        neck_params = []
        for n, p in model.named_parameters():
            if 'neck' in n and n not in param_names:
                neck_params.append(p)
                param_names.add(n)
        if neck_params:
            params.append({
                'params': neck_params,
                'lr': cls.OPTIMIZER['lr'] * cls.DIFFERENTIAL_LR['neck_lr_mult'],
                'name': 'neck'
            })
        
        # Head参数（最高学习率）
        head_params = []
        for n, p in model.named_parameters():
            if 'head' in n and n not in param_names:
                head_params.append(p)
                param_names.add(n)
        if head_params:
            params.append({
                'params': head_params,
                'lr': cls.OPTIMIZER['lr'] * cls.DIFFERENTIAL_LR['head_lr_mult'],
                'name': 'head'
            })
        
        # 其他参数
        other_params = []
        for n, p in model.named_parameters():
            if n not in param_names:
                other_params.append(p)
        if other_params:
            params.append({
                'params': other_params,
                'lr': cls.OPTIMIZER['lr'],
                'name': 'other'
            })
        
        optimizer_type = cls.OPTIMIZER['type']
        optimizer_cls = getattr(optim, optimizer_type)
        
        return optimizer_cls(
            params,
            weight_decay=cls.OPTIMIZER['weight_decay'],
            betas=cls.OPTIMIZER.get('betas', (0.9, 0.999)),
            eps=cls.OPTIMIZER.get('eps', 1e-8)
        )
    
    @classmethod
    def get_scheduler(cls, optimizer):
        """Get learning rate scheduler"""
        scheduler_type = cls.SCHEDULER['type']
        
        if scheduler_type == 'CosineAnnealingLR':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                optimizer,
                T_max=cls.SCHEDULER.get('T_max', cls.EPOCHS),
                eta_min=cls.SCHEDULER.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'StepLR':
            from torch.optim.lr_scheduler import StepLR
            return StepLR(
                optimizer,
                step_size=cls.SCHEDULER.get('step_size', 30),
                gamma=cls.SCHEDULER.get('gamma', 0.1)
            )
        elif scheduler_type == 'MultiStepLR':
            from torch.optim.lr_scheduler import MultiStepLR
            return MultiStepLR(
                optimizer,
                milestones=cls.SCHEDULER.get('milestones', [60, 90, 120]),
                gamma=cls.SCHEDULER.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure output directories exist"""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print configuration summary"""
        print("\n" + "="*80)
        print("Configuration Summary (Optimized)")
        print("="*80)
        print(f"Device: {cls.DEVICE}")
        print(f"Classes: {cls.CLASSES}")
        print(f"Num Classes: {cls.NUM_CLASSES}")
        print(f"Batch Size: {cls.BATCH_SIZE} (reduced for stability)")
        print(f"Epochs: {cls.EPOCHS} (increased for better convergence)")
        print(f"Image Size: {cls.IMG_SIZE}")
        print(f"Optimizer: {cls.OPTIMIZER['type']} (lr={cls.OPTIMIZER['lr']:.2e}, reduced)")
        print(f"Scheduler: {cls.SCHEDULER['type']}")
        print(f"Mixed Precision: {cls.USE_AMP}")
        print(f"Warmup Epochs: {cls.WARMUP_EPOCHS} (increased)")
        print(f"Grad Clip: {cls.GRAD_CLIP_MAX_NORM} (reduced)")
        print("="*80)
    
    @classmethod
    def print_config_enhanced(cls):
        """Print enhanced configuration summary"""
        cls.print_config()
        
        print("\n" + "="*80)
        print("Enhancement Modules Configuration (Optimized)")
        print("="*80)
        print(f"Attention Enabled: {cls.ATTENTION_CONFIG['use_attention']}")
        if cls.ATTENTION_CONFIG['use_attention']:
            print(f"  Types: {cls.ATTENTION_CONFIG['attention_types']} (streamlined)")
        print(f"Frequency Enabled: {cls.FREQUENCY_CONFIG['use_frequency']}")
        if cls.FREQUENCY_CONFIG['use_frequency']:
            print(f"  Types: {cls.FREQUENCY_CONFIG['frequency_types']} (streamlined)")
        print(f"Differential LR: {cls.DIFFERENTIAL_LR['enabled']}")
        if cls.DIFFERENTIAL_LR['enabled']:
            print(f"  Backbone: {cls.DIFFERENTIAL_LR['backbone_lr_mult']}x")
            print(f"  Attention: {cls.DIFFERENTIAL_LR['attention_lr_mult']}x (increased)")
            print(f"  Frequency: {cls.DIFFERENTIAL_LR['frequency_lr_mult']}x (increased)")
            print(f"  Neck: {cls.DIFFERENTIAL_LR['neck_lr_mult']}x (increased)")
            print(f"  Head: {cls.DIFFERENTIAL_LR['head_lr_mult']}x (increased)")
        
        print("\n" + "="*80)
        print("Optimization Highlights")
        print("="*80)
        print("✅ Lower learning rate (3e-5) for stable convergence")
        print("✅ Increased warmup (15 epochs) to prevent early instability")
        print("✅ Reduced batch size (4) to fit memory and improve gradient quality")
        print("✅ Enhanced class weights for small objects (fishing: 2.0x)")
        print("✅ Streamlined modules to reduce overfitting")
        print("✅ Differential LR: Head gets 2.0x, Backbone gets 0.1x")
        print("✅ More aggressive focal loss (alpha=0.30, gamma=2.5)")
        print("✅ Stricter anchor matching (pos_iou=0.6, neg_iou=0.3)")
        print("="*80)


# Global config instance
cfg = SARShipConfig()