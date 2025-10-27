"""
SAR Ship Detection Web Application - 6 Classes Version
6类别SAR舰船检测Web界面
"""

import gradio as gr
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config.sar_ship_config import cfg
from models.detectors.denodet_enhanced import EnhancedDenoDet


# 全局变量
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path, use_attention, use_frequency):
    """加载模型"""
    global MODEL
    
    try:
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return "❌ 请提供有效的checkpoint路径"
        
        print(f"📥 加载模型: {checkpoint_path}")
        
        MODEL = EnhancedDenoDet(
            use_attention=use_attention,
            use_frequency=use_frequency
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.to(DEVICE)
        MODEL.eval()
        
        epoch = checkpoint.get('epoch', 'Unknown')
        best_loss = checkpoint.get('best_loss', 'Unknown')
        
        info = f"✅ 模型加载成功!\n"
        info += f"Checkpoint: {Path(checkpoint_path).name}\n"
        info += f"Epoch: {epoch}\n"
        info += f"Best Loss: {best_loss:.4f}\n" if isinstance(best_loss, float) else f"Best Loss: {best_loss}\n"
        info += f"设备: {DEVICE}\n"
        info += f"类别数: {cfg.NUM_CLASSES}\n"
        info += f"类别: {', '.join(cfg.CLASSES)}"
        
        return info
        
    except Exception as e:
        error = f"❌ 加载失败: {str(e)}"
        print(error)
        import traceback
        traceback.print_exc()
        return error


def detect_ships(image, conf_threshold):
    """检测6类别舰船"""
    global MODEL
    
    if MODEL is None:
        return None, "⚠️ 请先加载模型!"
    
    if image is None:
        return None, "⚠️ 请上传图像"
    
    try:
        # 预处理
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        resized = cv2.resize(gray, cfg.IMG_SIZE)
        normalized = resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # 推理
        with torch.no_grad():
            inputs = {'img': img_tensor}
            results = MODEL(inputs, mode='test')[0]
        
        # 获取结果
        scores = results['scores'].cpu().numpy()
        bboxes = results['bboxes'].cpu().numpy()
        labels = results['labels'].cpu().numpy()
        
        # 过滤
        keep = scores >= conf_threshold
        scores = scores[keep]
        bboxes = bboxes[keep]
        labels = labels[keep]
        
        # 可视化
        vis_image = visualize_results(resized, bboxes, scores, labels)
        
        # 统计信息（按类别分组）
        class_counts = {cls: 0 for cls in cfg.CLASSES}
        for label in labels:
            class_counts[cfg.CLASSES[int(label)]] += 1
        
        info = f"🎯 检测完成!\n"
        info += f"检测到目标总数: {len(scores)}\n"
        
        if len(scores) > 0:
            info += f"最高置信度: {scores.max():.3f}\n"
            info += f"平均置信度: {scores.mean():.3f}\n"
            info += f"\n📊 各类别统计:\n"
            for class_name, count in class_counts.items():
                if count > 0:
                    info += f"  {class_name:<12} {count:>3} 个\n"
            
            info += f"\n📋 详细列表:\n"
            for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
                x1, y1, x2, y2 = bbox.astype(int)
                w, h = x2 - x1, y2 - y1
                class_name = cfg.CLASSES[int(label)]
                info += f"  {i+1}. {class_name}: 置信度={score:.3f}, 位置=({x1},{y1}), 大小={w}x{h}\n"
        
        return vis_image, info
        
    except Exception as e:
        error = f"❌ 检测失败: {str(e)}"
        print(error)
        import traceback
        traceback.print_exc()
        return None, error


def visualize_results(image, bboxes, scores, labels):
    """可视化6类别检测结果"""
    # 转换为BGR
    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()
    
    # 绘制边界框
    for bbox, score, label in zip(bboxes, scores, labels):
        x1, y1, x2, y2 = bbox.astype(int)
        
        # 根据类别选择颜色
        class_name = cfg.CLASSES[int(label)]
        color_bgr = cfg.CLASS_COLORS.get(class_name, (0, 255, 0))
        
        # 绘制矩形
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color_bgr, 2)
        
        # 绘制标签
        label_text = f"{class_name}: {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color_bgr, -1)
        cv2.putText(vis_img, label_text, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 添加统计和图例
    cv2.putText(vis_img, f"Total: {len(scores)}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 添加类别图例
    H = vis_img.shape[0]
    legend_y = H - 20 * len(cfg.CLASSES) - 10
    for i, class_name in enumerate(cfg.CLASSES):
        color = cfg.CLASS_COLORS[class_name]
        y_pos = legend_y + i * 20
        cv2.rectangle(vis_img, (10, y_pos-10), (30, y_pos+5), color, -1)
        cv2.putText(vis_img, class_name, (35, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 转换回RGB
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    return vis_img


# 创建界面
with gr.Blocks(title="SAR Ship Detection - 6 Classes") as demo:
    gr.Markdown("""
    # 🚢 SAR舰船检测系统 - 6类别版本
    ### Multi-Class SAR Ship Detection with Enhanced Deep Learning
    
    支持检测6种类型的舰船：货船、油轮、集装箱船、渔船、客船、军舰
    """)
    
    with gr.Tab("🎯 检测"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📤 上传SAR图像")
                input_image = gr.Image(label="输入图像", type="numpy")
                
                gr.Markdown("### ⚙️ 参数设置")
                conf_slider = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="置信度阈值")
                
                detect_btn = gr.Button("🔍 开始检测", variant="primary")
            
            with gr.Column():
                gr.Markdown("### 📊 检测结果")
                output_image = gr.Image(label="检测结果", type="numpy")
                detection_info = gr.Textbox(label="检测信息", lines=15)
        
        detect_btn.click(
            fn=detect_ships,
            inputs=[input_image, conf_slider],
            outputs=[output_image, detection_info]
        )
    
    with gr.Tab("⚙️ 模型设置"):
        gr.Markdown("### 📥 加载模型")
        
        checkpoint_path = gr.Textbox(
            label="模型路径",
            value="checkpoints/best_model.pth",
            placeholder="输入checkpoint路径..."
        )
        
        with gr.Row():
            use_attention = gr.Checkbox(label="使用Attention模块", value=True)
            use_frequency = gr.Checkbox(label="使用Frequency模块", value=True)
        
        load_btn = gr.Button("📥 加载模型", variant="primary")
        model_info = gr.Textbox(label="模型信息", lines=8)
        
        load_btn.click(
            fn=load_model,
            inputs=[checkpoint_path, use_attention, use_frequency],
            outputs=model_info
        )
        
        gr.Markdown("""
        ### 💡 使用说明
        
        **6类别说明：**
        - 🟢 **cargo** (货船): 一般货物运输船
        - 🔵 **tanker** (油轮): 液体货物运输船
        - 🟡 **container** (集装箱船): 集装箱运输船
        - 🟣 **fishing** (渔船): 小型捕鱼船只
        - 🟠 **passenger** (客船): 客运船只
        - 🟤 **military** (军舰): 军用舰艇
        
        **使用步骤：**
        1. 加载模型：输入checkpoint路径，点击"加载模型"
        2. 上传图像：切换到"检测"标签，上传SAR图像
        3. 调整阈值：根据需要调整置信度阈值（推荐0.3-0.5）
        4. 查看结果：点击"开始检测"，查看各类别检测结果
        
        **结果解读：**
        - 不同颜色代表不同类别
        - 检测信息会显示每个类别的数量统计
        - 详细列表显示每个目标的类别、置信度和位置
        """)
    
    with gr.Tab("📖 关于"):
        gr.Markdown(f"""
        ### 🚢 SAR舰船检测系统 - 6类别版本
        
        **版本**: 2.0.0 (Multi-Class)  
        **模型**: Enhanced DenoDet with 6-Class Support
        
        #### 🎯 检测类别
        
        | 类别 | 英文 | 特点 |
        |------|------|------|
        | 货船 | Cargo | 大型、结构规整 |
        | 油轮 | Tanker | 大型、圆柱形舱体 |
        | 集装箱船 | Container | 甲板有集装箱堆叠 |
        | 渔船 | Fishing | 小型、设备简单 |
        | 客船 | Passenger | 多层甲板、结构复杂 |
        | 军舰 | Military | 装备武器系统 |
        
        #### 📊 预期性能
        
        - **总体mAP@0.5**: 74-79%
        - **推理速度**: 25-30 FPS
        - **模型大小**: ~330MB
        - **参数量**: ~87M
        
        #### 🔧 技术特性
        
        - ✅ 频域增强模块（FFT、Gabor、小波）
        - ✅ 多种注意力机制（CBAM、散射注意力）
        - ✅ 差异化学习率优化
        - ✅ Per-class颜色编码
        - ✅ 类别统计和分析
        
        #### 📝 数据格式
        
        **YOLO格式标签：**
        ```
        class_id x_center y_center width height
        
        类别ID映射：
        0 = cargo
        1 = tanker
        2 = container
        3 = fishing
        4 = passenger
        5 = military
        ```
        
        #### 🎓 训练建议
        
        - 每类至少2000+样本
        - 训练120 epochs
        - 使用类别权重处理不平衡
        - 针对小目标（渔船）增加样本
        
        ---
        
        © 2024 SAR Ship Detection System - 6 Classes Version
        """)


if __name__ == "__main__":
    print("="*80)
    print("🚢 SAR Ship Detection - 6 Classes Web Interface")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Classes: {', '.join(cfg.CLASSES)}")
    print(f"Image Size: {cfg.IMG_SIZE}")
    print("="*80)
    
    # 启动服务
    try:
        demo.launch(
            share=True,  # 创建公网链接
            inbrowser=False,
            server_name="127.0.0.1",
            server_port=7860
        )
    except Exception as e:
        print(f"\n❌ Launch with 127.0.0.1 failed, trying share mode...")
        demo.launch(
            share=True,
            inbrowser=False
        )