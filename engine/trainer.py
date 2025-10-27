import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# 添加matplotlib支持
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

from config.sar_ship_config import cfg


class Trainer:
    """Complete training engine with validation, early stopping and visualization"""
    
    def __init__(self, model, train_loader, val_loader=None, experiment_name=None):
        self.model = model
        self.device = cfg.DEVICE
        self.epochs = cfg.EPOCHS
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 实验名称
        self.experiment_name = experiment_name or "default_experiment"
        
        # 📊 训练历史记录
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_cls_loss': [],
            'train_bbox_loss': [],
            'val_loss': [],
            'val_cls_loss': [],
            'val_bbox_loss': [],
            'learning_rate': []
        }
        
        # 📁 创建输出目录
        self.exp_dir = cfg.LOG_DIR / self.experiment_name
        self.curves_dir = self.exp_dir / 'curves'
        self.log_file = self.exp_dir / 'training.log'
        
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.curves_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Experiment directory: {self.exp_dir}")
        print(f"📈 Curves will be saved to: {self.curves_dir}")
        
        # 使用配置文件创建优化器
        print(f"🔧 Initializing optimizer: {cfg.OPTIMIZER['type']}")
        print(f"   Learning rate: {cfg.OPTIMIZER['lr']:.2e}")
        print(f"   Weight decay: {cfg.OPTIMIZER['weight_decay']:.2e}")
        
        self.optimizer = cfg.get_optimizer(model)
        
        # 使用配置文件创建调度器
        self.scheduler = cfg.get_scheduler(self.optimizer)
        
        # AMP - 从配置读取
        self.use_amp = cfg.USE_AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # 梯度裁剪阈值 - 从配置读取
        self.grad_clip_max_norm = cfg.GRAD_CLIP_MAX_NORM
        
        # Early stopping - 从配置读取
        self.best_loss = float('inf')
        self.patience = cfg.EARLY_STOP_PATIENCE
        self.patience_counter = 0
        
        # 日志间隔 - 从配置读取
        self.log_interval = cfg.LOG_INTERVAL
        
        # 调试模式 - 从配置读取
        self.detect_anomaly = cfg.DEBUG.get('detect_anomaly', False)
        
        # Warmup - 从配置读取
        self.warmup_epochs = cfg.WARMUP_EPOCHS
        
        # Ensure directories
        cfg.ensure_dirs()
        
        print(f"✅ Trainer initialized")
        print(f"   Train batches: {len(train_loader)}/epoch")
        if val_loader:
            print(f"   Val batches: {len(val_loader)}/epoch")
        print(f"   Mixed Precision: {self.use_amp}")
        print(f"   Gradient Clipping: {self.grad_clip_max_norm}")
        print(f"   Early Stop Patience: {self.patience}")
    
    def _warmup_lr(self, epoch):
        """Learning rate warmup"""
        if epoch <= self.warmup_epochs:
            warmup_factor = epoch / max(self.warmup_epochs, 1)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cfg.OPTIMIZER['lr'] * warmup_factor
    
    def _log_to_file(self, message):
        """Write to log file"""
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_bbox_loss = 0
        valid_batches = 0
        
        # Warmup学习率
        if epoch <= self.warmup_epochs:
            self._warmup_lr(epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        
        for batch_idx, data in enumerate(pbar):
            imgs = data['img'].to(self.device)
            gt_bboxes = [bbox.to(self.device) for bbox in data['gt_bboxes']]
            gt_labels = [label.to(self.device) for label in data['gt_labels']]
            
            inputs = {
                'img': imgs,
                'gt_bboxes': gt_bboxes,
                'gt_labels': gt_labels
            }
            
            # Forward with optional AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs, mode='train')
                    loss = outputs['total_loss']
                    cls_loss = outputs['cls_loss']
                    bbox_loss = outputs['bbox_loss']
            else:
                outputs = self.model(inputs, mode='train')
                loss = outputs['total_loss']
                cls_loss = outputs['cls_loss']
                bbox_loss = outputs['bbox_loss']
            
            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0.0:
                if batch_idx % 100 == 0 or batch_idx < 5:
                    msg = f"⚠️ Warning: Invalid loss at batch {batch_idx}: {loss.item()}"
                    print(f"\n{msg}")
                    self._log_to_file(msg)
                continue
            
            # Backward
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.grad_clip_max_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.grad_clip_max_norm
                )
                self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_bbox_loss += bbox_loss.item()
            valid_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cls': f"{cls_loss.item():.4f}",
                'bbox': f"{bbox_loss.item():.4f}"
            })
        
        # 计算平均损失
        if valid_batches == 0:
            print("⚠️ Warning: No valid training batches")
            return {'loss': float('inf'), 'cls_loss': 0, 'bbox_loss': 0}
        
        avg_loss = total_loss / valid_batches
        avg_cls_loss = total_cls_loss / valid_batches
        avg_bbox_loss = total_bbox_loss / valid_batches
        
        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'bbox_loss': avg_bbox_loss
        }
    
    def validate(self):
        """Validate model"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        total_cls_loss = 0
        total_bbox_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            for data in tqdm(self.val_loader, desc="Validating"):
                imgs = data['img'].to(self.device)
                gt_bboxes = [bbox.to(self.device) for bbox in data['gt_bboxes']]
                gt_labels = [label.to(self.device) for label in data['gt_labels']]
                
                inputs = {
                    'img': imgs,
                    'gt_bboxes': gt_bboxes,
                    'gt_labels': gt_labels
                }
                
                outputs = self.model(inputs, mode='train')
                loss = outputs['total_loss']
                cls_loss = outputs['cls_loss']
                bbox_loss = outputs['bbox_loss']
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    total_cls_loss += cls_loss.item()
                    total_bbox_loss += bbox_loss.item()
                    valid_batches += 1
        
        if valid_batches == 0:
            print("⚠️ Warning: No valid validation batches")
            return None
        
        avg_loss = total_loss / valid_batches
        avg_cls_loss = total_cls_loss / valid_batches
        avg_bbox_loss = total_bbox_loss / valid_batches
        
        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'bbox_loss': avg_bbox_loss
        }
    
    def plot_curves(self):
        """绘制训练曲线"""
        if len(self.history['epoch']) == 0:
            return
        
        epochs = self.history['epoch']
        
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 总损失
        axes[0, 0].plot(epochs, self.history['train_loss'], 
                       'b-o', linewidth=2, markersize=4, label='Train Loss')
        if len(self.history['val_loss']) > 0:
            val_epochs = [e for e in epochs if e % cfg.VAL_INTERVAL == 0]
            val_epochs = val_epochs[:len(self.history['val_loss'])]
            axes[0, 0].plot(val_epochs, self.history['val_loss'], 
                           'r-s', linewidth=2, markersize=5, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 分类损失
        axes[0, 1].plot(epochs, self.history['train_cls_loss'], 
                       'b-o', linewidth=2, markersize=4, label='Train Cls Loss')
        if len(self.history['val_cls_loss']) > 0:
            axes[0, 1].plot(val_epochs, self.history['val_cls_loss'], 
                           'r-s', linewidth=2, markersize=5, label='Val Cls Loss')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Classification Loss', fontsize=12)
        axes[0, 1].set_title('Classification Loss', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 回归损失
        axes[1, 0].plot(epochs, self.history['train_bbox_loss'], 
                       'b-o', linewidth=2, markersize=4, label='Train BBox Loss')
        if len(self.history['val_bbox_loss']) > 0:
            axes[1, 0].plot(val_epochs, self.history['val_bbox_loss'], 
                           'r-s', linewidth=2, markersize=5, label='Val BBox Loss')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('BBox Loss', fontsize=12)
        axes[1, 0].set_title('Bounding Box Loss', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 学习率
        if len(self.history['learning_rate']) > 0:
            axes[1, 1].plot(epochs, self.history['learning_rate'], 
                           'g-o', linewidth=2, markersize=4)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        save_path = self.curves_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training curves saved to: {save_path}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'config': {
                'lr': cfg.OPTIMIZER['lr'],
                'batch_size': cfg.BATCH_SIZE,
                'img_size': cfg.IMG_SIZE
            }
        }
        
        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存常规checkpoint
        ckpt_path = cfg.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = cfg.CHECKPOINT_DIR / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"🌟 Saved best model: loss={self.best_loss:.4f}")
        
        # 保存最新模型
        latest_path = cfg.CHECKPOINT_DIR / 'latest.pth'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        print(f"📥 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        # 恢复训练历史
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"   Best loss: {self.best_loss:.4f}")
        
        return checkpoint['epoch']
    
    def train(self, resume_from=None):
        """Main training loop"""
        start_epoch = 1
        
        # 断点续训
        if resume_from:
            if resume_from == 'latest':
                resume_from = cfg.CHECKPOINT_DIR / 'latest.pth'
            start_epoch = self.load_checkpoint(resume_from) + 1
        
        # 打印训练信息
        print("="*80)
        print("🚀 Starting Training")
        print("="*80)
        cfg.print_config()
        print(f"📊 Train samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"📊 Val samples: {len(self.val_loader.dataset)}")
        print(f"🔄 Training from epoch {start_epoch} to {self.epochs}")
        print("="*80)
        
        # 写入日志
        self._log_to_file("="*80)
        self._log_to_file(f"Experiment: {self.experiment_name}")
        self._log_to_file(f"Start time: {self.exp_dir.name}")
        self._log_to_file("="*80)
        
        # 启用异常检测（第一个epoch）
        if start_epoch == 1 and self.detect_anomaly:
            print("🔍 Enabling anomaly detection for first epoch...")
            torch.autograd.set_detect_anomaly(True)
        
        for epoch in range(start_epoch, self.epochs + 1):
            # 第一个epoch后关闭异常检测
            if epoch == 2 and self.detect_anomaly:
                print("✅ Disabling anomaly detection (training is stable)")
                torch.autograd.set_detect_anomaly(False)
            
            # Train
            try:
                train_metrics = self.train_epoch(epoch)
            except RuntimeError as e:
                print(f"\n❌ Runtime error at epoch {epoch}: {e}")
                print("💡 Tip: Try reducing batch size or learning rate")
                self.save_checkpoint(epoch)
                raise
            
            # 更新学习率
            if epoch > self.warmup_epochs:
                self.scheduler.step()
            
            # Validate
            val_metrics = None
            if self.val_loader and epoch % cfg.VAL_INTERVAL == 0:
                val_metrics = self.validate()
            
            # 记录到history
            lr = self.optimizer.param_groups[0]['lr']
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_cls_loss'].append(train_metrics['cls_loss'])
            self.history['train_bbox_loss'].append(train_metrics['bbox_loss'])
            self.history['learning_rate'].append(lr)
            
            if val_metrics is not None:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_cls_loss'].append(val_metrics['cls_loss'])
                self.history['val_bbox_loss'].append(val_metrics['bbox_loss'])
            
            # Log
            log_msg = (f"Epoch {epoch}/{self.epochs} - "
                      f"train_loss: {train_metrics['loss']:.4f} "
                      f"(cls: {train_metrics['cls_loss']:.4f}, "
                      f"bbox: {train_metrics['bbox_loss']:.4f}), "
                      f"lr: {lr:.2e}")
            
            if val_metrics is not None:
                log_msg += (f", val_loss: {val_metrics['loss']:.4f} "
                          f"(cls: {val_metrics['cls_loss']:.4f}, "
                          f"bbox: {val_metrics['bbox_loss']:.4f})")
            
            print(log_msg)
            self._log_to_file(log_msg)
            
            # 每5个epoch绘制曲线
            if epoch % 5 == 0:
                self.plot_curves()
            
            # Save checkpoint
            if epoch % cfg.SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            monitor_loss = val_metrics['loss'] if val_metrics else train_metrics['loss']
            
            if monitor_loss < self.best_loss:
                self.best_loss = monitor_loss
                self.save_checkpoint(epoch, is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    print(f"   No improvement for {self.patience} epochs")
                    break
        
        # 训练结束，绘制最终曲线
        self.plot_curves()
        
        print("="*80)
        print(f"✅ Training completed!")
        print(f"   Best loss: {self.best_loss:.4f}")
        print(f"   Checkpoints: {cfg.CHECKPOINT_DIR}")
        print(f"   Curves: {self.curves_dir}")
        print("="*80)
        
        return self.best_loss