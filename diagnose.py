"""
6类别配置完整验证脚本
运行此脚本检查所有配置是否正确，是否可以开始训练
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import traceback


def print_section(title):
    """打印章节标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def check_config():
    """检查配置文件"""
    print_section("1️⃣ 检查配置文件")
    
    errors = []
    warnings = []
    
    try:
        from config.sar_ship_config import cfg
        print("✅ 配置文件导入成功")
        
        # 检查类别定义
        print(f"\n📋 类别配置:")
        print(f"   CLASSES = {cfg.CLASSES}")
        print(f"   NUM_CLASSES = {cfg.NUM_CLASSES if hasattr(cfg, 'NUM_CLASSES') else 'NOT DEFINED'}")
        
        if len(cfg.CLASSES) != 6:
            errors.append(f"❌ cfg.CLASSES 应该有6个元素，实际: {len(cfg.CLASSES)}")
        else:
            print(f"   ✅ 类别数量正确: {len(cfg.CLASSES)}")
        
        expected_classes = ['cargo', 'tanker', 'container', 'fishing', 'passenger', 'military']
        if cfg.CLASSES != expected_classes:
            warnings.append(f"⚠️  类别名称与预期不同")
            print(f"   预期: {expected_classes}")
            print(f"   实际: {cfg.CLASSES}")
        
        # 检查HEAD配置
        print(f"\n📋 HEAD配置:")
        print(f"   num_classes = {cfg.HEAD.get('num_classes', 'NOT DEFINED')}")
        
        if cfg.HEAD.get('num_classes', 1) != 6:
            errors.append(f"❌ cfg.HEAD['num_classes'] 应该是6，实际: {cfg.HEAD.get('num_classes')}")
        else:
            print(f"   ✅ HEAD num_classes 正确: 6")
        
        # 检查类别颜色
        if hasattr(cfg, 'CLASS_COLORS'):
            print(f"\n📋 类别颜色:")
            if len(cfg.CLASS_COLORS) == 6:
                print(f"   ✅ 已配置 {len(cfg.CLASS_COLORS)} 种颜色")
                for cls, color in cfg.CLASS_COLORS.items():
                    print(f"      {cls:<12} {color}")
            else:
                warnings.append(f"⚠️  CLASS_COLORS 应该有6种颜色，实际: {len(cfg.CLASS_COLORS)}")
        else:
            warnings.append("⚠️  未定义 CLASS_COLORS（建议添加）")
        
        # 检查训练参数
        print(f"\n📋 训练参数:")
        print(f"   EPOCHS = {cfg.EPOCHS}")
        print(f"   BATCH_SIZE = {cfg.BATCH_SIZE}")
        print(f"   Learning Rate = {cfg.OPTIMIZER['lr']:.2e}")
        print(f"   WARMUP_EPOCHS = {cfg.WARMUP_EPOCHS}")
        
        if cfg.EPOCHS < 100:
            warnings.append(f"⚠️  6类别建议训练至少100 epochs，当前: {cfg.EPOCHS}")
        
        return errors, warnings
        
    except Exception as e:
        errors.append(f"❌ 配置文件错误: {str(e)}")
        traceback.print_exc()
        return errors, warnings


def check_model():
    """检查模型初始化"""
    print_section("2️⃣ 检查模型初始化")
    
    errors = []
    
    try:
        from models.detectors.denodet_enhanced import EnhancedDenoDet
        from config.sar_ship_config import cfg
        
        print("正在初始化模型...")
        model = EnhancedDenoDet(use_attention=True, use_frequency=True)
        print("✅ 模型初始化成功")
        
        # 检查模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📊 模型参数:")
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # 检查Head的类别数
        if hasattr(model, 'head') and hasattr(model.head, 'num_classes'):
            head_classes = model.head.num_classes
            print(f"\n📋 Head配置:")
            print(f"   num_classes = {head_classes}")
            
            if head_classes != 6:
                errors.append(f"❌ model.head.num_classes 应该是6，实际: {head_classes}")
            else:
                print(f"   ✅ Head类别数正确: 6")
        
        # 测试前向传播
        print(f"\n🧪 测试前向传播...")
        model.eval()
        with torch.no_grad():
            # 创建假数据
            dummy_img = torch.randn(1, 1, 512, 512)
            dummy_inputs = {'img': dummy_img}
            
            try:
                results = model(dummy_inputs, mode='test')
                print(f"   ✅ 推理模式成功")
                print(f"   输出格式: {results[0].keys()}")
                
                # 检查输出
                labels = results[0]['labels']
                if len(labels) > 0:
                    max_label = labels.max().item()
                    if max_label >= 6:
                        errors.append(f"❌ 输出的label超出范围 (0-5): {max_label}")
                    else:
                        print(f"   ✅ 标签范围正确: 0-{max_label}")
                
            except Exception as e:
                errors.append(f"❌ 前向传播失败: {str(e)}")
                traceback.print_exc()
        
        return errors
        
    except Exception as e:
        errors.append(f"❌ 模型初始化失败: {str(e)}")
        traceback.print_exc()
        return errors


def check_data():
    """检查数据加载"""
    print_section("3️⃣ 检查数据加载")
    
    errors = []
    warnings = []
    
    try:
        from data.loaders import get_train_dataloader
        from config.sar_ship_config import cfg
        
        print("正在加载训练数据...")
        
        # 检查数据路径
        print(f"\n📁 数据路径:")
        print(f"   训练图像: {cfg.TRAIN_IMG_DIR}")
        print(f"   训练标签: {cfg.TRAIN_LABEL_DIR}")
        
        if not cfg.TRAIN_IMG_DIR.exists():
            errors.append(f"❌ 训练图像目录不存在: {cfg.TRAIN_IMG_DIR}")
            return errors, warnings
        
        if not cfg.TRAIN_LABEL_DIR.exists():
            errors.append(f"❌ 训练标签目录不存在: {cfg.TRAIN_LABEL_DIR}")
            return errors, warnings
        
        # 统计文件数量
        num_images = len(list(cfg.TRAIN_IMG_DIR.glob('*.jpg'))) + \
                    len(list(cfg.TRAIN_IMG_DIR.glob('*.png')))
        num_labels = len(list(cfg.TRAIN_LABEL_DIR.glob('*.txt')))
        
        print(f"   图像数量: {num_images}")
        print(f"   标签数量: {num_labels}")
        
        if num_images == 0:
            errors.append(f"❌ 未找到训练图像")
            return errors, warnings
        
        if num_labels == 0:
            errors.append(f"❌ 未找到训练标签")
            return errors, warnings
        
        if abs(num_images - num_labels) > 0:
            warnings.append(f"⚠️  图像和标签数量不匹配: {num_images} vs {num_labels}")
        
        # 加载一个batch
        print(f"\n🧪 测试数据加载...")
        loader = get_train_dataloader()
        print(f"   ✅ DataLoader创建成功")
        print(f"   Batch数量: {len(loader)}")
        
        try:
            batch = next(iter(loader))
            print(f"   ✅ 成功加载一个batch")
            
            # 检查batch内容
            print(f"\n📦 Batch内容:")
            print(f"   图像shape: {batch['img'].shape}")
            print(f"   GT boxes数量: {[len(b) for b in batch['gt_bboxes'][:3]]}")
            print(f"   GT labels数量: {[len(l) for l in batch['gt_labels'][:3]]}")
            
            # 检查标签范围
            all_labels = torch.cat(batch['gt_labels'])
            if len(all_labels) > 0:
                min_label = all_labels.min().item()
                max_label = all_labels.max().item()
                print(f"\n📋 标签统计:")
                print(f"   标签范围: {min_label} - {max_label}")
                
                if min_label < 0 or max_label > 5:
                    errors.append(f"❌ 标签超出范围 (应该是0-5): {min_label}-{max_label}")
                else:
                    print(f"   ✅ 标签范围正确 (0-5)")
                
                # 统计各类别数量
                print(f"\n📊 类别分布:")
                for i in range(6):
                    count = (all_labels == i).sum().item()
                    if count > 0:
                        class_name = cfg.CLASSES[i]
                        print(f"   {i}: {class_name:<12} {count:>3} 个")
                
                # 检查是否有未使用的类别
                used_classes = set(all_labels.unique().cpu().numpy())
                for i in range(6):
                    if i not in used_classes:
                        warnings.append(f"⚠️  类别 {i} ({cfg.CLASSES[i]}) 在此batch中未出现")
            
        except Exception as e:
            errors.append(f"❌ 数据加载失败: {str(e)}")
            traceback.print_exc()
        
        return errors, warnings
        
    except Exception as e:
        errors.append(f"❌ 数据检查失败: {str(e)}")
        traceback.print_exc()
        return errors, warnings


def check_optimizer():
    """检查优化器"""
    print_section("4️⃣ 检查优化器")
    
    errors = []
    
    try:
        from models.detectors.denodet_enhanced import EnhancedDenoDet
        from config.sar_ship_config import cfg
        
        model = EnhancedDenoDet(use_attention=False, use_frequency=False)
        
        # 标准优化器
        print("🔧 标准优化器:")
        opt_std = cfg.get_optimizer(model)
        print(f"   类型: {type(opt_std).__name__}")
        print(f"   参数组数: {len(opt_std.param_groups)}")
        print(f"   学习率: {opt_std.param_groups[0]['lr']:.2e}")
        
        # 增强优化器
        print(f"\n🔧 增强优化器 (差异化学习率):")
        model_enh = EnhancedDenoDet(use_attention=True, use_frequency=True)
        opt_enh = cfg.get_optimizer_enhanced(model_enh)
        print(f"   类型: {type(opt_enh).__name__}")
        print(f"   参数组数: {len(opt_enh.param_groups)}")
        
        for i, group in enumerate(opt_enh.param_groups):
            num_params = sum(p.numel() for p in group['params'])
            print(f"   组{i}: {num_params:>10,} 参数, lr={group['lr']:.2e}")
        
        print(f"   ✅ 优化器创建成功")
        
        return errors
        
    except Exception as e:
        errors.append(f"❌ 优化器检查失败: {str(e)}")
        traceback.print_exc()
        return errors


def check_training():
    """检查训练流程"""
    print_section("5️⃣ 检查训练流程")
    
    errors = []
    
    try:
        from models.detectors.denodet_enhanced import EnhancedDenoDet
        from data.loaders import get_train_dataloader
        from config.sar_ship_config import cfg
        
        print("🔧 创建模型...")
        model = EnhancedDenoDet(use_attention=True, use_frequency=True)
        device = cfg.DEVICE
        model = model.to(device)
        
        print("📦 加载数据...")
        loader = get_train_dataloader()
        batch = next(iter(loader))
        
        print("🧪 测试训练模式...")
        model.train()
        
        imgs = batch['img'].to(device)
        gt_bboxes = [bbox.to(device) for bbox in batch['gt_bboxes']]
        gt_labels = [label.to(device) for label in batch['gt_labels']]
        
        inputs = {
            'img': imgs,
            'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels
        }
        
        try:
            outputs = model(inputs, mode='train')
            print(f"   ✅ 前向传播成功")
            print(f"   Total loss: {outputs['total_loss'].item():.4f}")
            print(f"   Cls loss: {outputs['cls_loss'].item():.4f}")
            print(f"   Bbox loss: {outputs['bbox_loss'].item():.4f}")
            
            # 测试反向传播
            print(f"\n🔙 测试反向传播...")
            outputs['total_loss'].backward()
            print(f"   ✅ 反向传播成功")
            
            # 检查梯度
            has_grad = False
            nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    has_grad = True
                    if torch.isnan(param.grad).any():
                        nan_grad = True
                        break
            
            if not has_grad:
                errors.append("❌ 没有计算梯度")
            elif nan_grad:
                errors.append("❌ 梯度包含NaN")
            else:
                print(f"   ✅ 梯度正常")
            
        except Exception as e:
            errors.append(f"❌ 训练流程失败: {str(e)}")
            traceback.print_exc()
        
        return errors
        
    except Exception as e:
        errors.append(f"❌ 训练检查失败: {str(e)}")
        traceback.print_exc()
        return errors


def print_summary(all_errors, all_warnings):
    """打印总结"""
    print_section("📊 验证总结")
    
    # 打印警告
    if all_warnings:
        print(f"\n⚠️  警告 ({len(all_warnings)} 个):")
        for warning in all_warnings:
            print(f"   {warning}")
    
    # 打印错误
    if all_errors:
        print(f"\n❌ 错误 ({len(all_errors)} 个):")
        for error in all_errors:
            print(f"   {error}")
        
        print("\n" + "="*80)
        print("❌ 验证失败！请修复上述错误后再训练。")
        print("="*80)
        return False
    else:
        if all_warnings:
            print("\n" + "="*80)
            print("⚠️  验证通过，但有一些警告建议处理。")
            print("可以开始训练，但建议先查看警告信息。")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("✅ 所有检查通过！可以开始训练了！")
            print("="*80)
            print("\n🚀 开始训练命令:")
            print("   python train_enhanced.py --batch_size 4 --epochs 120 --exp_name my_6class")
            print("\n或使用baseline模式（更快）:")
            print("   python train_enhanced.py --baseline --batch_size 8 --epochs 100")
            print("="*80)
        return True


def main():
    """主函数"""
    print("="*80)
    print("🔍 SAR舰船检测 - 6类别配置验证")
    print("="*80)
    print("此脚本将检查所有配置是否正确，确保可以开始训练")
    print("="*80)
    
    all_errors = []
    all_warnings = []
    
    # 1. 检查配置
    errors, warnings = check_config()
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    
    if errors:
        print("\n❌ 配置文件有错误，无法继续检查")
        print_summary(all_errors, all_warnings)
        return False
    
    # 2. 检查模型
    errors = check_model()
    all_errors.extend(errors)
    
    if errors:
        print("\n❌ 模型初始化有错误，无法继续检查")
        print_summary(all_errors, all_warnings)
        return False
    
    # 3. 检查数据
    errors, warnings = check_data()
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    
    if errors:
        print("\n❌ 数据加载有错误，无法继续检查")
        print_summary(all_errors, all_warnings)
        return False
    
    # 4. 检查优化器
    errors = check_optimizer()
    all_errors.extend(errors)
    
    # 5. 检查训练流程
    errors = check_training()
    all_errors.extend(errors)
    
    # 打印总结
    success = print_summary(all_errors, all_warnings)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)