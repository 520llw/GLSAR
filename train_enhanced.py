"""
Enhanced Training Script for SAR Ship Detection
支持Attention和Frequency模块的训练脚本
"""

import torch
import argparse
import random
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config.sar_ship_config import cfg
from data.loaders import get_train_dataloader, get_val_dataloader
from models.detectors.denodet_enhanced import EnhancedDenoDet
from engine.trainer import Trainer


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EnhancedTrainer(Trainer):
    """Enhanced trainer with differential learning rates"""
    
    def __init__(self, model, train_loader, val_loader=None, 
                 experiment_name=None, use_differential_lr=True):
        
        self.use_differential_lr = use_differential_lr
        
        # 调用父类初始化（会创建标准优化器）
        super().__init__(model, train_loader, val_loader, experiment_name)
        
        # 🔥 如果使用差异化学习率，重新创建优化器
        if use_differential_lr:
            print("🔥 Using differential learning rates for enhanced modules")
            self.optimizer = cfg.get_optimizer_enhanced(model)
            self.scheduler = cfg.get_scheduler(self.optimizer)
            
            # 打印参数组信息
            print("\n📊 Parameter Groups:")
            for i, param_group in enumerate(self.optimizer.param_groups):
                num_params = sum(p.numel() for p in param_group['params'])
                print(f"  Group {i}: {num_params:,} params, lr={param_group['lr']:.2e}")


def main():
    parser = argparse.ArgumentParser(description="SAR Ship Detection Training - Enhanced")
    
    # 基础训练参数
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--no_val', action='store_true',
                       help='Disable validation during training')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name')
    
    # 🔥 增强模块控制参数
    parser.add_argument('--no_attention', action='store_true',
                       help='Disable attention modules')
    parser.add_argument('--no_frequency', action='store_true',
                       help='Disable frequency modules')
    parser.add_argument('--baseline', action='store_true',
                       help='Train baseline model without any enhancements')
    parser.add_argument('--no_diff_lr', action='store_true',
                       help='Disable differential learning rates')
    
    args = parser.parse_args()
    
    # Override config if needed
    if args.batch_size != cfg.BATCH_SIZE:
        cfg.BATCH_SIZE = args.batch_size
    if args.epochs != cfg.EPOCHS:
        cfg.EPOCHS = args.epochs
    
    # 🔥 设置增强模块开关
    use_attention = not args.no_attention and not args.baseline
    use_frequency = not args.no_frequency and not args.baseline
    use_diff_lr = not args.no_diff_lr and not args.baseline
    
    # 🔥 生成实验名称（包含模块信息）
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        modules = []
        if use_attention:
            modules.append("att")
        if use_frequency:
            modules.append("freq")
        if use_diff_lr:
            modules.append("difflr")
        if not modules:
            modules.append("baseline")
        module_str = "_".join(modules)
        experiment_name = f"sar_{module_str}_{timestamp}"
    else:
        experiment_name = args.exp_name
    
    # Setup
    print("="*80)
    print("SAR Ship Detection Training - Enhanced Version")
    print("="*80)
    print(f"📝 Experiment: {experiment_name}")
    print(f"🖥️  Device: {cfg.DEVICE}")
    print(f"🔄 Epochs: {cfg.EPOCHS}")
    print(f"📦 Batch Size: {cfg.BATCH_SIZE}")
    print(f"🖼️  Image Size: {cfg.IMG_SIZE}")
    print(f"🌱 Seed: {cfg.SEED}")
    print(f"✨ Attention Modules: {'✅' if use_attention else '❌'}")
    print(f"📡 Frequency Modules: {'✅' if use_frequency else '❌'}")
    print(f"🎓 Differential LR: {'✅' if use_diff_lr else '❌'}")
    print("="*80)
    
    set_random_seed(cfg.SEED)
    cfg.ensure_dirs()
    
    # 🔥 Model with enhancement modules
    print("\n🔧 Initializing model...")
    try:
        model = EnhancedDenoDet(
            use_attention=use_attention,
            use_frequency=use_frequency
        ).to(cfg.DEVICE)
    except Exception as e:
        print(f"\n❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")
    
    # Data
    print("\n📦 Loading datasets...")
    train_loader = get_train_dataloader()
    val_loader = None if args.no_val else get_val_dataloader()
    
    print(f"✅ Train batches: {len(train_loader)}")
    if val_loader:
        print(f"✅ Val batches: {len(val_loader)}")
    
    # 🔥 Trainer with enhanced optimizer
    print("\n🎯 Creating trainer...")
    trainer = EnhancedTrainer(
        model, 
        train_loader, 
        val_loader,
        experiment_name=experiment_name,
        use_differential_lr=use_diff_lr
    )
    
    # Train
    print("\n" + "="*80)
    print("🚀 Starting Training...")
    print("="*80)
    
    try:
        trainer.train(resume_from=args.resume)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("💾 Saving checkpoint...")
        trainer.save_checkpoint(trainer.history['epoch'][-1] if trainer.history['epoch'] else 0)
        return
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("="*80)
    print("✅ Training completed successfully!")
    print("="*80)
    
    # 结果位置
    print("\n📊 Results saved to:")
    print(f"   📈 Training curves: logs/{experiment_name}/curves/training_curves.png")
    print(f"   📝 Training log: logs/{experiment_name}/training.log")
    print(f"   💾 Checkpoints: checkpoints/")
    print(f"   🌟 Best model: checkpoints/best_model.pth")
    
    # 🔥 性能分析提示
    if use_attention or use_frequency:
        print("\n💡 Analysis Tools:")
        print("   To analyze module contributions:")
        print("   python tools/analyze_modules.py --checkpoint checkpoints/best_model.pth")
        print("\n   To visualize predictions:")
        print("   python tools/visualize.py --checkpoint checkpoints/best_model.pth")
    
    print()


if __name__ == "__main__":
    main()