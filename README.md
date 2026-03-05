# 城市治理问题智能识别系统

基于深度学习的城市治理问题自动检测系统，支持违章停车、占道经营等问题的实时识别。

## 📋 功能特性

- ✅ **实时视频检测** - 支持摄像头实时监测
- ✅ **图像批量检测** - 支持单张/批量图像分析
- ✅ **Web 演示界面** - 开箱即用的 Gradio Web UI
- ✅ **违规记录保存** - 自动保存违规证据和统计
- ✅ **多模型支持** - YOLOv8/v9/v10, DETR
- ✅ **易于扩展** - 支持自定义类别和数据集

## 🎯 检测类别

| ID | 类别 | 说明 |
|----|------|------|
| 0 | illegal_parking | 违章停车 |
| 1 | street_vendor | 占道经营 |
| 2 | illegal_stall | 违规摊位 |
| 3 | blocked_passage | 堵塞通道 |
| 4 | illegal_advertisement | 违规广告 |
| 5 | garbage_dumping | 乱倒垃圾 |
| 6 | damaged_facility | 设施损坏 |

## 🚀 快速开始

### 1. 安装依赖

```bash
cd urban-governance-ai
pip install -r requirements.txt
```

### 2. 准备数据集

#### 方式 A: 使用已有数据集
将标注好的数据按以下结构组织：
```
data/urban_violations/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

#### 方式 B: 转换现有标注格式
```bash
# VOC 格式转 YOLO
python scripts/prepare_data.py --task voc2yolo --input /path/to/VOC --output data/urban_violations/labels

# COCO 格式转 YOLO
python scripts/prepare_data.py --task coco2yolo --input /path/to/coco.json --output data/urban_violations/labels

# 划分数据集
python scripts/prepare_data.py --task split --input data/raw_images --output data/urban_violations
```

### 3. 训练模型

```bash
# 使用默认配置训练
python scripts/train.py

# 或指定配置文件
python scripts/train.py --config configs/train_config.yaml
```

### 4. 推理检测

#### 摄像头实时检测
```bash
python scripts/inference.py --source camera --input 0
```

#### 图像检测
```bash
python scripts/inference.py --source image --input path/to/image.jpg
```

#### 视频文件检测
```bash
python scripts/inference.py --source video --input path/to/video.mp4 --save-video
```

### 5. Web Demo

```bash
python scripts/web_demo.py --port 7860
```

然后在浏览器访问 `http://localhost:7860`

## 📁 项目结构

```
urban-governance-ai/
├── configs/                  # 配置文件
│   ├── dataset_config.yaml   # 数据集配置
│   └── train_config.yaml     # 训练参数配置
├── data/                     # 数据目录
│   ├── images/               # 图像数据
│   └── labels/               # 标注文件
├── models/                   # 模型输出
│   └── outputs/              # 训练输出
├── scripts/                  # 脚本
│   ├── prepare_data.py       # 数据准备
│   ├── train.py              # 模型训练
│   ├── inference.py          # 实时推理
│   └── web_demo.py           # Web 演示
├── utils/                    # 工具函数
├── inference/                # 推理结果
│   └── results/              # 检测结果保存
├── requirements.txt          # 依赖
└── README.md                 # 说明文档
```

## 🔧 自定义配置

### 修改检测类别
编辑 `configs/dataset_config.yaml`:
```yaml
names:
  0: your_class_1
  1: your_class_2
  # ...
```

### 调整训练参数
编辑 `configs/train_config.yaml`:
```yaml
epochs: 100        # 训练轮数
batch: 16          # 批次大小
imgsz: 640         # 输入尺寸
lr0: 0.01          # 初始学习率
```

## 📊 模型导出

```bash
# 导出为 ONNX
python scripts/train.py --export models/outputs/.../best.pt --export-format onnx

# 导出为 TensorRT (需要 NVIDIA GPU)
python scripts/train.py --export models/outputs/.../best.pt --export-format engine
```

## 🛠️ 常见问题

### Q: 数据集太小怎么办？
A: 建议使用数据增强（mosaic, mixup），并减少 batch size。100-200 张标注图像即可开始训练。

### Q: 检测精度不够？
A:
1. 增加训练轮数 (epochs)
2. 使用更大的模型 (yolov8m/l/x)
3. 增加高质量标注数据
4. 调整置信度阈值

### Q: 检测速度太慢？
A:
1. 使用更小的模型 (yolov8n)
2. 降低输入分辨率 (imgsz=416)
3. 导出为 ONNX/TensorRT 加速
4. 使用 GPU 推理

## 📝 许可证

MIT License
