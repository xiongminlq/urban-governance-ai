# 数据标注指南

## 📋 概述

本指南介绍如何准备和标注城市治理问题检测的数据集。

## 🎯 检测类别

| ID | 类别名称 | 中文说明 | 示例场景 |
|----|----------|----------|----------|
| 0 | illegal_parking | 违章停车 | 车辆停放在禁停区域、占用消防通道 |
| 1 | street_vendor | 占道经营 | 流动摊贩占用道路经营 |
| 2 | illegal_stall | 违规摊位 | 固定摊位超出经营范围 |
| 3 | blocked_passage | 堵塞通道 | 物品堵塞人行通道、消防通道 |
| 4 | illegal_advertisement | 违规广告 | 乱张贴广告、横幅 |
| 5 | garbage_dumping | 乱倒垃圾 | 随意堆放垃圾 |
| 6 | damaged_facility | 设施损坏 | 损坏的公共设施 |

## 🛠️ 标注工具推荐

### 1. LabelImg (推荐)
**安装**:
```bash
pip install labelimg
labelImg
```

**使用步骤**:
1. 打开 LabelImg
2. 选择 `Open Dir` 加载图像目录
3. 选择 `Change Save Dir` 设置标注保存目录
4. 按 `W` 键绘制矩形框
5. 选择对应类别
6. 按 `Ctrl+S` 保存
7. 按 `D` 键切换到下一张图

**输出格式**: 选择 **YOLO** 格式

### 2. CVAT (在线标注)
网址：https://cvat.ai/

适合团队协作，支持视频标注。

### 3. Label Studio
**安装**:
```bash
pip install label-studio
label-studio start
```

功能强大，支持多种数据类型。

## 📁 数据集结构

标注完成后的数据集结构：

```
data/urban_violations/
├── images/
│   ├── train/          # 训练图像 (约 80%)
│   ├── val/            # 验证图像 (约 15%)
│   └── test/           # 测试图像 (约 5%)
└── labels/
    ├── train/          # 训练标注
    ├── val/            # 验证标注
    └── test/           # 测试标注
```

## 📝 YOLO 标注格式

每个标注文件 (.txt) 的格式：
```
<class_id> <x_center> <y_center> <width> <height>
```

**说明**:
- `class_id`: 类别 ID (0, 1, 2, ...)
- `x_center`: 边界框中心 X 坐标 / 图像宽度
- `y_center`: 边界框中心 Y 坐标 / 图像高度
- `width`: 边界框宽度 / 图像宽度
- `height`: 边界框高度 / 图像高度

**示例**:
```
0 0.523438 0.631250 0.156250 0.218750
1 0.312500 0.453125 0.093750 0.156250
```

## 💡 标注技巧

### 1. 边界框要紧贴目标
✅ 正确：边界框紧密包围目标
❌ 错误：边界框过大或过小

### 2. 遮挡目标也要标注
即使目标被部分遮挡，也要标注可见部分

### 3. 小目标处理
对于过小的目标（小于 32x32 像素），可以选择性标注

### 4. 模糊目标处理
无法明确识别的目标不标注

### 5. 类别模糊处理
当目标同时属于多个类别时，选择最符合的主要类别

## 📊 数据集建议

### 最小可用数据集
- 每个类别至少 **50 张** 图像
- 总计 **300+ 张** 标注图像
- 可以训练出可用的模型

### 推荐数据集
- 每个类别 **200+ 张** 图像
- 总计 **1000+ 张** 标注图像
- 可以训练出高精度的模型

### 理想数据集
- 每个类别 **500+ 张** 图像
- 总计 **5000+ 张** 标注图像
- 可以训练出生产级模型

## 🔍 数据增强

在训练配置中已包含以下增强策略：

| 增强方式 | 参数 | 说明 |
|----------|------|------|
| Mosaic | mosaic=1.0 | 四图拼接，强烈推荐 |
| 水平翻转 | fliplr=0.5 | 50% 概率翻转 |
| HSV 调色 | hsv_h=0.015 | 颜色扰动 |
| 平移 | translate=0.1 | 10% 平移 |
| 缩放 | scale=0.5 | 随机缩放 |

## 🚀 快速开始标注

1. **采集图像**
   ```bash
   python scripts/collect_data.py --task capture --count 100
   ```

2. **启动 LabelImg**
   ```bash
   pip install labelimg
   labelImg
   ```

3. **划分数据集**
   ```bash
   python scripts/prepare_data.py --task split --input data/collected_data --output data/urban_violations
   ```

4. **生成配置文件**
   ```bash
   python scripts/prepare_data.py --task generate_config --output configs/dataset_config.yaml
   ```

## ⚠️ 注意事项

1. **隐私保护**: 标注时注意模糊人脸、车牌等敏感信息
2. **质量控制**: 定期检查标注质量，修正错误标注
3. **版本管理**: 使用不同版本号管理数据集迭代
4. **备份**: 定期备份标注数据

## 📞 获取帮助

如有问题，请查阅：
- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [LabelImg GitHub](https://github.com/HumanSignal/labelImg)
