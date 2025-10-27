#!/usr/bin/env python3
"""
自动检测数据位置并修复配置文件
"""

from pathlib import Path
import shutil

# 项目根目录
PROJECT_ROOT = Path('/home/gjw/Groksar_rebuild')
CONFIG_FILE = PROJECT_ROOT / 'config' / 'sar_ship_config.py'

print("="*80)
print("🔧 自动修复配置文件")
print("="*80)

# ============================================================================
# 第一步：搜索数据集
# ============================================================================
print("\n1️⃣ 搜索数据集...")

possible_paths = [
    Path('/data/store/SARdataset'),
    PROJECT_ROOT / 'data' / 'data1',
    PROJECT_ROOT / 'data' / 'data2',
    PROJECT_ROOT / 'data',
]

found_data = None

for base_path in possible_paths:
    print(f"   检查: {base_path}")
    
    if not base_path.exists():
        print(f"      ❌ 不存在")
        continue
    
    # 检查标准结构: train/images
    train_img_dir = base_path / 'train' / 'images'
    train_label_dir = base_path / 'train' / 'labels'
    val_img_dir = base_path / 'val' / 'images'
    val_label_dir = base_path / 'val' / 'labels'
    
    if train_img_dir.exists():
        images = list(train_img_dir.glob('*.jpg')) + list(train_img_dir.glob('*.png'))
        if len(images) > 0:
            labels = list(train_label_dir.glob('*.txt')) if train_label_dir.exists() else []
            
            print(f"      ✅ 找到数据集！")
            print(f"         训练图像: {len(images)} 张")
            print(f"         训练标签: {len(labels)} 个")
            
            if val_img_dir.exists():
                val_images = list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.png'))
                val_labels = list(val_label_dir.glob('*.txt')) if val_label_dir.exists() else []
                print(f"         验证图像: {len(val_images)} 张")
                print(f"         验证标签: {len(val_labels)} 个")
            
            found_data = {
                'base_path': base_path,
                'train_img_dir': train_img_dir,
                'train_label_dir': train_label_dir,
                'val_img_dir': val_img_dir,
                'val_label_dir': val_label_dir,
                'num_train': len(images),
                'num_val': len(val_images) if val_img_dir.exists() else 0,
            }
            break

if not found_data:
    print("\n❌ 错误：未找到数据集！")
    print("\n请确保数据存在于以下位置之一：")
    for p in possible_paths:
        print(f"   - {p}/train/images/")
    exit(1)

print(f"\n✅ 使用数据集: {found_data['base_path']}")

# ============================================================================
# 第二步：备份配置文件
# ============================================================================
print("\n2️⃣ 备份配置文件...")

backup_file = CONFIG_FILE.parent / 'sar_ship_config.py.backup_auto'
if not backup_file.exists():
    shutil.copy(CONFIG_FILE, backup_file)
    print(f"   ✅ 已备份到: {backup_file}")
else:
    print(f"   ⚠️  备份已存在: {backup_file}")

# ============================================================================
# 第三步：读取并修改配置
# ============================================================================
print("\n3️⃣ 修改配置文件...")

with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到需要修改的行
new_lines = []
modified = False

for i, line in enumerate(lines):
    # 找到 DATA_ROOT 定义
    if 'DATA_ROOT = PROJECT_ROOT' in line and 'data' in line.lower():
        # 注释掉原来的行
        new_lines.append(f"    # {line.strip()}  # 原配置已注释\n")
        
        # 添加新的配置（使用相对路径）
        rel_path = found_data['base_path'].relative_to(PROJECT_ROOT)
        new_lines.append(f"    DATA_ROOT = PROJECT_ROOT / '{rel_path}'\n")
        modified = True
        print(f"   ✅ 修改 DATA_ROOT = PROJECT_ROOT / '{rel_path}'")
    
    # 保持 TRAIN_IMG_DIR 等的相对路径不变
    elif any(x in line for x in ['TRAIN_IMG_DIR', 'TRAIN_LABEL_DIR', 'VAL_IMG_DIR', 'VAL_LABEL_DIR']) and '=' in line:
        new_lines.append(line)
    else:
        new_lines.append(line)

# 如果没有找到DATA_ROOT，在合适位置插入
if not modified:
    print("   ⚠️  未找到 DATA_ROOT 配置，添加新配置...")
    
    # 找到 # Data paths 注释后插入
    for i, line in enumerate(new_lines):
        if '# Data paths' in line or 'DATA_ROOT' in line:
            # 插入新配置
            rel_path = found_data['base_path'].relative_to(PROJECT_ROOT)
            insert_lines = [
                f"\n",
                f"    # 数据路径配置（自动修复）\n",
                f"    DATA_ROOT = PROJECT_ROOT / '{rel_path}'\n",
            ]
            new_lines = new_lines[:i+1] + insert_lines + new_lines[i+1:]
            print(f"   ✅ 添加 DATA_ROOT = PROJECT_ROOT / '{rel_path}'")
            break

# ============================================================================
# 第四步：写入修改后的配置
# ============================================================================
with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"   ✅ 配置文件已更新")

# ============================================================================
# 第五步：验证配置
# ============================================================================
print("\n4️⃣ 验证配置...")

# 重新导入配置
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# 清除缓存
if 'config.sar_ship_config' in sys.modules:
    del sys.modules['config.sar_ship_config']

try:
    from config.sar_ship_config import cfg
    
    print(f"   TRAIN_IMG_DIR: {cfg.TRAIN_IMG_DIR}")
    print(f"   TRAIN_LABEL_DIR: {cfg.TRAIN_LABEL_DIR}")
    print(f"   VAL_IMG_DIR: {cfg.VAL_IMG_DIR}")
    print(f"   VAL_LABEL_DIR: {cfg.VAL_LABEL_DIR}")
    
    # 检查路径是否存在
    if cfg.TRAIN_IMG_DIR.exists():
        num_train_img = len(list(cfg.TRAIN_IMG_DIR.glob('*.jpg'))) + len(list(cfg.TRAIN_IMG_DIR.glob('*.png')))
        print(f"\n   ✅ 训练图像: {num_train_img} 张")
    else:
        print(f"\n   ❌ 训练图像目录不存在: {cfg.TRAIN_IMG_DIR}")
    
    if cfg.TRAIN_LABEL_DIR.exists():
        num_train_label = len(list(cfg.TRAIN_LABEL_DIR.glob('*.txt')))
        print(f"   ✅ 训练标签: {num_train_label} 个")
    else:
        print(f"   ❌ 训练标签目录不存在: {cfg.TRAIN_LABEL_DIR}")
    
    if cfg.VAL_IMG_DIR.exists():
        num_val_img = len(list(cfg.VAL_IMG_DIR.glob('*.jpg'))) + len(list(cfg.VAL_IMG_DIR.glob('*.png')))
        print(f"   ✅ 验证图像: {num_val_img} 张")
    else:
        print(f"   ⚠️  验证图像目录不存在: {cfg.VAL_IMG_DIR}")
    
    if cfg.VAL_LABEL_DIR.exists():
        num_val_label = len(list(cfg.VAL_LABEL_DIR.glob('*.txt')))
        print(f"   ✅ 验证标签: {num_val_label} 个")
    else:
        print(f"   ⚠️  验证标签目录不存在: {cfg.VAL_LABEL_DIR}")

except Exception as e:
    print(f"\n   ❌ 配置验证失败: {e}")
    print(f"   请手动检查配置文件: {CONFIG_FILE}")
    exit(1)

# ============================================================================
# 完成
# ============================================================================
print("\n" + "="*80)
print("✅ 配置文件修复完成！")
print("="*80)
print("\n📝 修改摘要:")
print(f"   配置文件: {CONFIG_FILE}")
print(f"   备份文件: {backup_file}")
print(f"   数据位置: {found_data['base_path']}")
print(f"   训练样本: {found_data['num_train']} 张")
print(f"   验证样本: {found_data['num_val']} 张")

print("\n🚀 下一步:")
print("   1. 运行诊断验证: python diagnose.py")
print("   2. 开始训练: python train_enhanced.py --no_frequency --batch_size 4 --epochs 150 --exp_name production")

print("\n💡 如需恢复原配置:")
print(f"   cp {backup_file} {CONFIG_FILE}")
print()