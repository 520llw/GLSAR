import os
import random
import shutil

def split_dataset(
    raw_image_dir,
    raw_annot_dir,
    output_root,
    img_suffix=".jpg",  # 图像文件后缀
    split_ratio=(0.7, 0.2, 0.1),  # 训练:验证:测试 比例
    random_seed=42  # 随机种子确保可复现
):
    """
    划分数据集为train/val/test，不依赖配置文件
    
    参数:
        raw_image_dir: 原始图像目录（绝对路径）
        raw_annot_dir: 原始标注目录（绝对路径）
        output_root: 划分后的数据保存根目录（绝对路径）
        img_suffix: 图像文件后缀（如.jpg/.png）
        split_ratio: 数据集划分比例
        random_seed: 随机种子，保证划分结果可复现
    """
    # 1. 验证输入路径是否存在
    if not os.path.exists(raw_image_dir):
        raise FileNotFoundError(f"原始图像目录不存在: {raw_image_dir}")
    if not os.path.exists(raw_annot_dir):
        raise FileNotFoundError(f"原始标注目录不存在: {raw_annot_dir}")
    
    # 2. 创建输出目录结构（train/val/test各自包含images和labels）
    for split in ["train", "val", "test"]:
        # 图像目录
        img_dir = os.path.join(output_root, split, "images")
        os.makedirs(img_dir, exist_ok=True)
        # 标注目录
        annot_dir = os.path.join(output_root, split, "labels")
        os.makedirs(annot_dir, exist_ok=True)
        print(f"已创建目录: {img_dir}")
        print(f"已创建目录: {annot_dir}")
    
    # 3. 获取所有图像文件（仅保留指定后缀的文件）
    all_images = [f for f in os.listdir(raw_image_dir) if f.endswith(img_suffix)]
    total_count = len(all_images)
    if total_count == 0:
        raise ValueError(f"在 {raw_image_dir} 中未找到 {img_suffix} 格式的图像文件")
    print(f"\n共发现 {total_count} 个 {img_suffix} 格式的图像文件")
    
    # 4. 随机打乱并按比例划分
    random.seed(random_seed)
    random.shuffle(all_images)
    
    # 计算各子集数量
    train_count = int(total_count * split_ratio[0])
    val_count = int(total_count * split_ratio[1])
    test_count = total_count - train_count - val_count  # 确保总数匹配
    
    # 划分图像列表
    train_images = all_images[:train_count]
    val_images = all_images[train_count : train_count + val_count]
    test_images = all_images[train_count + val_count :]
    
    print(f"\n划分比例: 训练集 {split_ratio[0]*100}% | 验证集 {split_ratio[1]*100}% | 测试集 {split_ratio[2]*100}%")
    print(f"划分数量: 训练集 {len(train_images)} | 验证集 {len(val_images)} | 测试集 {len(test_images)}")
    
    # 5. 复制文件到对应目录
    def copy_files(image_list, split_name):
        """复制图像和标注到指定划分目录"""
        for img_name in image_list:
            # 图像文件名（不含后缀）
            img_basename = os.path.splitext(img_name)[0]
            
            # 复制图像
            src_img = os.path.join(raw_image_dir, img_name)
            dst_img = os.path.join(output_root, split_name, "images", img_name)
            shutil.copyfile(src_img, dst_img)
            
            # 复制标注（假设标注文件为 .txt 格式，与图像同名）
            annot_name = f"{img_basename}.txt"
            src_annot = os.path.join(raw_annot_dir, annot_name)
            if os.path.exists(src_annot):
                dst_annot = os.path.join(output_root, split_name, "labels", annot_name)
                shutil.copyfile(src_annot, dst_annot)
            else:
                print(f"警告: 未找到 {img_name} 对应的标注文件 {annot_name}")
    
    # 分别复制训练集、验证集、测试集
    copy_files(train_images, "train")
    copy_files(val_images, "val")
    copy_files(test_images, "test")
    
    print(f"\n数据集划分完成！结果保存在: {output_root}")

if __name__ == "__main__":
    # 原始图像和标注目录（直接指定绝对路径）
    RAW_IMAGE_DIR = "/home/gjw/Groksar_rebuild/data/SAR-Ship-Dataset/ship_dataset_v0/images"
    RAW_ANNOT_DIR = "/home/gjw/Groksar_rebuild/data/SAR-Ship-Dataset/ship_dataset_v0/labels"
    
    # 划分后的数据保存目录（目标路径）
    OUTPUT_ROOT = "/home/gjw/Groksar_rebuild/data/SAR-Ship-Dataset/ship_dataset_v0"
    
    # 图像文件后缀（根据你的实际图像格式修改，如.png）
    IMG_SUFFIX = ".jpg"
    
    # 执行划分（70%训练，20%验证，10%测试）
    split_dataset(
        raw_image_dir=RAW_IMAGE_DIR,
        raw_annot_dir=RAW_ANNOT_DIR,
        output_root=OUTPUT_ROOT,
        img_suffix=IMG_SUFFIX,
        split_ratio=(0.7, 0.2, 0.1),
        random_seed=42
    )
