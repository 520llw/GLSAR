"""
完整的训练日志和可视化系统
支持TensorBoard、文本日志、图像保存
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from datetime import datetime
import json


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir, experiment_name=None):
        """
        Args:
            log_dir: 日志根目录
            experiment_name: 实验名称（如果为None，使用时间戳）
        """
        self.log_dir = Path(log_dir)
        
        # 创建实验目录
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.exp_dir = self.log_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self.tb_dir = self.exp_dir / "tensorboard"
        self.vis_dir = self.exp_dir / "visualizations"
        self.curve_dir = self.exp_dir / "curves"
        self.log_file = self.exp_dir / "training.log"
        
        self.tb_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)
        self.curve_dir.mkdir(exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(str(self.tb_dir))
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_cls_loss': [],
            'train_bbox_loss': [],
            'val_loss': [],
            'val_cls_loss': [],
            'val_bbox_loss': [],
            'learning_rate': [],
            'epoch': []
        }
        
        print(f"📊 Logger initialized")
        print(f"   Experiment: {experiment_name}")
        print(f"   Log dir: {self.exp_dir}")
        print(f"   TensorBoard: tensorboard --logdir={self.tb_dir}")
    
    def log_metrics(self, epoch, metrics, phase='train'):
        """
        记录训练指标
        
        Args:
            epoch: 当前epoch
            metrics: 指标字典 {'loss': ..., 'cls_loss': ..., 'bbox_loss': ...}
            phase: 'train' 或 'val'
        """
        # TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{key}', value, epoch)
        
        # 保存到历史
        self.history['epoch'].append(epoch)
        if phase == 'train':
            self.history['train_loss'].append(metrics.get('loss', 0))
            self.history['train_cls_loss'].append(metrics.get('cls_loss', 0))
            self.history['train_bbox_loss'].append(metrics.get('bbox_loss', 0))
        else:
            self.history['val_loss'].append(metrics.get('loss', 0))
            self.history['val_cls_loss'].append(metrics.get('cls_loss', 0))
            self.history['val_bbox_loss'].append(metrics.get('bbox_loss', 0))
        
        # 写入文本日志
        log_msg = f"[Epoch {epoch}] {phase}: "
        log_msg += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self._write_log(log_msg)
    
    def log_learning_rate(self, epoch, lr):
        """记录学习率"""
        self.writer.add_scalar('train/learning_rate', lr, epoch)
        self.history['learning_rate'].append(lr)
    
    def log_images(self, epoch, images, predictions, targets, prefix='train'):
        """
        记录图像和检测结果
        
        Args:
            epoch: 当前epoch
            images: [B, C, H, W] tensor
            predictions: list of dicts with 'boxes', 'scores', 'labels'
            targets: list of dicts with 'boxes', 'labels'
            prefix: 'train' 或 'val'
        """
        # 只保存前4张图
        num_images = min(4, images.shape[0])
        
        fig, axes = plt.subplots(2, num_images, figsize=(num_images*4, 8))
        if num_images == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(num_images):
            # 原图 + GT
            img = images[i].cpu().numpy()
            if img.shape[0] == 1:  # 单通道
                img = img[0]
            else:
                img = img.transpose(1, 2, 0)
            
            # 归一化到[0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # 绘制GT
            axes[0, i].imshow(img, cmap='gray')
            if i < len(targets) and 'boxes' in targets[i]:
                boxes = targets[i]['boxes'].cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                        fill=False, color='green', linewidth=2)
                    axes[0, i].add_patch(rect)
            axes[0, i].set_title(f'GT Image {i}')
            axes[0, i].axis('off')
            
            # 绘制预测
            axes[1, i].imshow(img, cmap='gray')
            if i < len(predictions) and 'boxes' in predictions[i]:
                boxes = predictions[i]['boxes'].cpu().numpy()
                scores = predictions[i]['scores'].cpu().numpy()
                for box, score in zip(boxes, scores):
                    if score > 0.5:  # 只显示高置信度
                        x1, y1, x2, y2 = box
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                            fill=False, color='red', linewidth=2)
                        axes[1, i].add_patch(rect)
                        axes[1, i].text(x1, y1-5, f'{score:.2f}',
                                       color='red', fontsize=8,
                                       bbox=dict(facecolor='white', alpha=0.5))
            axes[1, i].set_title(f'Pred Image {i}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # 保存到文件
        save_path = self.vis_dir / f'{prefix}_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 记录到TensorBoard
        self.writer.add_figure(f'{prefix}/predictions', fig, epoch)
    
    def plot_curves(self, save=True):
        """
        绘制训练曲线
        
        Returns:
            fig: matplotlib figure对象
        """
        epochs = self.history['epoch']
        if len(epochs) == 0:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 总损失
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if len(self.history['val_loss']) > 0:
            val_epochs = epochs[::len(epochs)//len(self.history['val_loss']) or 1][:len(self.history['val_loss'])]
            axes[0, 0].plot(val_epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 分类损失
        axes[0, 1].plot(epochs, self.history['train_cls_loss'], 'b-', label='Train Cls Loss', linewidth=2)
        if len(self.history['val_cls_loss']) > 0:
            axes[0, 1].plot(val_epochs, self.history['val_cls_loss'], 'r-', label='Val Cls Loss', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Classification Loss', fontsize=12)
        axes[0, 1].set_title('Classification Loss', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 回归损失
        axes[1, 0].plot(epochs, self.history['train_bbox_loss'], 'b-', label='Train BBox Loss', linewidth=2)
        if len(self.history['val_bbox_loss']) > 0:
            axes[1, 0].plot(val_epochs, self.history['val_bbox_loss'], 'r-', label='Val BBox Loss', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('BBox Loss', fontsize=12)
        axes[1, 0].set_title('Bounding Box Loss', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 学习率
        if len(self.history['learning_rate']) > 0:
            axes[1, 1].plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.curve_dir / 'training_curves.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📈 Training curves saved to: {save_path}")
        
        return fig
    
    def save_history(self):
        """保存训练历史到JSON"""
        history_file = self.exp_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"💾 Training history saved to: {history_file}")
    
    def _write_log(self, message):
        """写入文本日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
    
    def close(self):
        """关闭logger"""
        self.writer.close()
        self.save_history()
        self.plot_curves(save=True)
        print(f"✅ Logger closed. Results saved to: {self.exp_dir}")


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    # 创建logger
    logger = TrainingLogger(
        log_dir='logs',
        experiment_name='sar_ship_pretrain'
    )
    
    # 模拟训练
    for epoch in range(1, 11):
        # 训练指标
        train_metrics = {
            'loss': 0.5 - epoch * 0.03,
            'cls_loss': 0.3 - epoch * 0.02,
            'bbox_loss': 0.2 - epoch * 0.01
        }
        logger.log_metrics(epoch, train_metrics, phase='train')
        
        # 验证指标
        if epoch % 2 == 0:
            val_metrics = {
                'loss': 0.45 - epoch * 0.025,
                'cls_loss': 0.28 - epoch * 0.018,
                'bbox_loss': 0.17 - epoch * 0.007
            }
            logger.log_metrics(epoch, val_metrics, phase='val')
        
        # 学习率
        lr = 1e-4 * (0.95 ** epoch)
        logger.log_learning_rate(epoch, lr)
    
    # 关闭
    logger.close()
    
    print("\n运行以下命令查看TensorBoard:")
    print(f"tensorboard --logdir=logs/sar_ship_pretrain/tensorboard")