#!/usr/bin/env python3
"""
模型训练脚本
支持 YOLOv8 和 DETR 模型训练
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO


def train_yolov8(config_path):
    """
    使用 YOLOv8 进行训练

    Args:
        config_path: 训练配置文件路径
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print("城市治理问题检测 - YOLOv8 训练")
    print("=" * 50)

    # 加载预训练模型
    model = YOLO(config.get('model', 'yolov8n.pt'))
    print(f"加载模型：{config.get('model')}")

    # 开始训练
    results = model.train(
        # 数据集配置
        data='../configs/dataset_config.yaml',

        # 训练参数
        epochs=config.get('epochs', 100),
        batch=config.get('batch', 16),
        imgsz=config.get('imgsz', 640),

        # 优化器
        optimizer=config.get('optimizer', 'SGD'),
        lr0=config.get('lr0', 0.01),
        lrf=config.get('lrf', 0.1),
        momentum=config.get('momentum', 0.937),
        weight_decay=config.get('weight_decay', 0.0005),
        warmup_epochs=config.get('warmup_epochs', 3),

        # 数据增强
        hsv_h=config.get('hsv_h', 0.015),
        hsv_s=config.get('hsv_s', 0.7),
        hsv_v=config.get('hsv_v', 0.4),
        degrees=config.get('degrees', 0.0),
        translate=config.get('translate', 0.1),
        scale=config.get('scale', 0.5),
        shear=config.get('shear', 0.0),
        perspective=config.get('perspective', 0.0),
        flipud=config.get('flipud', 0.0),
        fliplr=config.get('fliplr', 0.5),
        mosaic=config.get('mosaic', 1.0),
        mixup=config.get('mixup', 0.0),

        # 设备设置
        device=config.get('device', 0),
        workers=config.get('workers', 8),
        amp=config.get('amp', True),

        # 其他
        project=config.get('project', '../models/outputs'),
        name=config.get('name', 'urban_governance_yolov8'),
        exist_ok=config.get('exist_ok', False),
        patience=config.get('patience', 50),
        save=True,
        verbose=config.get('verbose', True),
        seed=config.get('seed', 42),
    )

    print("\n" + "=" * 50)
    print("训练完成!")
    print(f"最佳模型保存位置：{results.best_save_path}")
    print("=" * 50)

    return model, results


def export_model(model_path, format='onnx', imgsz=640):
    """
    导出模型用于部署

    Args:
        model_path: 模型路径
        format: 导出格式 (onnx, torchscript, engine)
        imgsz: 输入图像尺寸
    """
    model = YOLO(model_path)

    export_path = model.export(
        format=format,
        imgsz=imgsz
    )

    print(f"模型已导出：{export_path}")
    return export_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模型训练工具")
    parser.add_argument("--model", choices=["yolov8", "yolov9", "yolov10"], default="yolov8",
                        help="模型类型")
    parser.add_argument("--config", default="../configs/train_config.yaml",
                        help="训练配置文件路径")
    parser.add_argument("--export", help="导出模型路径")
    parser.add_argument("--export-format", default="onnx",
                        help="导出格式")

    args = parser.parse_args()

    if args.export:
        export_model(args.export, args.export_format)
    else:
        train_yolov8(args.config)
