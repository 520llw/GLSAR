# 🚢 SAR舰船检测Web界面使用说明

## 📦 安装

### 1. 安装Gradio
```bash
pip install gradio
```

### 2. 确认其他依赖
```bash
pip install torch torchvision opencv-python numpy
```

---

## 🚀 启动

### 方式1: 直接启动
```bash
python app.py
```

### 方式2: 自定义端口
```bash
# 修改 app.py 最后几行
demo.launch(
    server_port=8080,  # 改成你想要的端口
    share=True         # 如果需要公网访问
)
```

### 方式3: 后台运行
```bash
nohup python app.py > web_demo.log 2>&1 &
```

---

## 🎯 使用步骤

### 第一步：加载模型

1. 打开浏览器访问: `http://localhost:7860`
2. 点击 **"⚙️ 模型设置"** 标签
3. 输入checkpoint路径（默认: `checkpoints/best_model.pth`）
4. 选择模块配置:
   - ✅ **使用Attention模块** (推荐)
   - ✅ **使用Frequency模块** (推荐)
5. 点击 **"📥 加载模型"**

**成功提示:**
```
✅ 模型加载成功!
   Checkpoint: best_model.pth
   Epoch: 45
   Best Loss: 0.5432
   设备: cuda
   Attention: True
   Frequency: True
```

### 第二步：上传图像

1. 切换到 **"🎯 检测"** 标签
2. 点击上传区域或拖拽SAR图像
3. 支持格式: `.jpg`, `.png`, `.bmp`, `.tif`

**或者使用示例图像:**
- 点击下方的示例图像快速测试

### 第三步：调整参数

- **置信度阈值** (0.1-0.9):
  - 较低 (0.2-0.3): 检测更多目标，可能有误检
  - 中等 (0.3-0.5): **推荐**，平衡准确率和召回率
  - 较高 (0.5-0.7): 只保留高置信度目标

- **NMS阈值** (0.1-0.9):
  - 较低 (0.3-0.4): 抑制更多重叠框
  - 中等 (0.5): **推荐**
  - 较高 (0.6-0.7): 保留更多框

### 第四步：检测

1. 点击 **"🔍 开始检测"** 按钮
2. 等待检测完成（通常1-3秒）
3. 查看结果:
   - **左侧**: 原始图像
   - **右侧**: 检测结果可视化
   - **下方**: 详细检测信息表格

---

## 📊 结果解读

### 颜色编码

| 颜色 | 置信度范围 | 含义 |
|------|-----------|------|
| 🟢 绿色 | > 0.7 | 高置信度 - 确定是舰船 |
| 🟡 黄色 | 0.5 - 0.7 | 中置信度 - 很可能是舰船 |
| 🟠 橙色 | < 0.5 | 低置信度 - 可能是舰船 |

### 检测信息

```
🎯 检测完成!
   检测到目标数: 3
   最高置信度: 0.876
   平均置信度: 0.654
   置信度阈值: 0.3

📊 检测详情:
   目标1: 置信度=0.876, 位置=(234,156), 大小=45x32
   目标2: 置信度=0.721, 位置=(389,201), 大小=38x28
   目标3: 置信度=0.465, 位置=(512,345), 大小=28x22
```

### 详细表格

| 目标ID | 置信度 | 中心位置 | 大小 | 边界框 |
|--------|--------|----------|------|--------|
| 1 | 0.876 | (234, 156) | 45 × 32 | (211, 140, 256, 172) |
| 2 | 0.721 | (389, 201) | 38 × 28 | (370, 187, 408, 215) |
| 3 | 0.465 | (512, 345) | 28 × 22 | (498, 334, 526, 356) |

---

## 🔧 高级配置

### 修改默认参数

编辑 `app.py` 中的参数:

```python
# 默认置信度阈值
conf_slider = gr.Slider(
    value=0.3,  # 改成你想要的默认值
    ...
)

# 默认checkpoint路径
checkpoint_path = gr.Textbox(
    value="checkpoints/best_model.pth",  # 改成你的路径
    ...
)
```

### 添加更多示例图像

```python
gr.Examples(
    examples=[
        ["data/val/images/example1.jpg"],
        ["data/val/images/example2.jpg"],
        ["path/to/your/example3.jpg"],  # 添加更多
    ],
    ...
)
```

### 修改界面主题

```python
gr.Blocks(theme=gr.themes.Soft())  # 可选: Soft, Base, Glass, Monochrome
```

---

## 🌐 远程访问

### 创建公网链接（临时）

```python
demo.launch(
    share=True  # 会生成一个公网URL
)
```

**输出示例:**
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxxx.gradio.live
```

### 配置持久化服务

使用nginx反向代理:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

---

## 📱 移动端访问

1. 确保手机和电脑在同一局域网
2. 启动时使用:
```python
demo.launch(server_name="0.0.0.0")
```
3. 在手机浏览器访问: `http://你的电脑IP:7860`

---

## ⚠️ 常见问题

### Q1: 模型加载失败
```
❌ 模型加载失败: No such file or directory
```
**解决**: 检查checkpoint路径是否正确
```bash
ls checkpoints/best_model.pth
```

### Q2: CUDA out of memory
```
❌ 检测失败: CUDA out of memory
```
**解决**: 在 `app.py` 中强制使用CPU
```python
self.device = torch.device('cpu')  # 改成CPU
```

### Q3: 端口被占用
```
OSError: [Errno 48] Address already in use
```
**解决**: 更改端口或杀掉占用进程
```bash
# 查看占用端口的进程
lsof -i :7860

# 杀掉进程
kill -9 <PID>

# 或使用其他端口
demo.launch(server_port=8080)
```

### Q4: 图像无法上传
**解决**: 检查图像格式和大小
- 支持格式: JPG, PNG, BMP, TIF
- 建议大小: < 10MB

### Q5: 检测速度慢
**原因**: 
- CPU模式较慢
- 图像过大
- 模型参数量大

**优化**:
```bash
# 使用baseline模型（参数更少）
python app.py --baseline

# 或减小输入图像尺寸
# 修改 config/sar_ship_config.py
IMG_SIZE = (256, 256)  # 原来是(512, 512)
```

---

## 🎨 界面截图

### 检测界面
```
+----------------------------------+----------------------------------+
|          上传SAR图像              |          检测结果                |
|                                  |                                  |
|   [拖拽或点击上传]                |   [检测结果可视化]               |
|                                  |                                  |
|                                  |   🎯 检测完成!                   |
|   ⚙️ 检测参数                    |      检测到目标数: 3              |
|   置信度阈值: [====●=====] 0.3   |      最高置信度: 0.876           |
|   NMS阈值:    [======●===] 0.5   |      平均置信度: 0.654           |
|                                  |                                  |
|   [🔍 开始检测]                   |                                  |
+----------------------------------+----------------------------------+
|                      详细检测结果表格                                |
|  目标ID | 置信度 | 中心位置 | 大小 | 边界框                       |
+-----------------------------------------------------------------------+
```

---

## 📞 技术支持

遇到问题？

1. 查看日志: `web_demo.log`
2. 提交Issue: [GitHub Issues]
3. 联系邮箱: your.email@example.com

---

## 🎉 快速开始命令汇总

```bash
# 1. 安装依赖
pip install gradio torch opencv-python numpy

# 2. 启动Web界面
python app.py

# 3. 打开浏览器
# 访问 http://localhost:7860

# 4. 加载模型
# 在"模型设置"标签输入: checkpoints/best_model.pth

# 5. 上传图像并检测
# 在"检测"标签上传SAR图像

# 🎊 完成！
```

---

© 2024 SAR Ship Detection System