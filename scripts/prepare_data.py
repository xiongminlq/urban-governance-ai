#!/usr/bin/env python3
"""
数据准备脚本
- 从不同标注格式转换为 YOLO 格式
- 划分训练集/验证集
- 生成数据集配置文件
"""

import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def split_dataset(images_dir, output_dir, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05):
    """
    将图像数据集划分为训练集、验证集和测试集

    Args:
        images_dir: 原始图像目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须为 1"

    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    images = [f for f in os.listdir(images_dir)
              if Path(f).suffix.lower() in image_extensions]

    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:]
    }

    # 创建目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # 复制图像文件
    for split, files in splits.items():
        print(f"\n处理 {split} 集 ({len(files)} 张图像)...")
        for img_file in tqdm(files):
            src_path = os.path.join(images_dir, img_file)
            dst_path = os.path.join(output_dir, 'images', split, img_file)
            shutil.copy2(src_path, dst_path)

    print(f"\n数据集划分完成:")
    print(f"  训练集：{len(splits['train'])} 张")
    print(f"  验证集：{len(splits['val'])} 张")
    print(f"  测试集：{len(splits['test'])} 张")

    return splits


def convert_voc_to_yolo(voc_dir, output_dir, class_mapping):
    """
    将 Pascal VOC 格式转换为 YOLO 格式

    Args:
        voc_dir: VOC 格式目录 (包含 Annotations 和 Images)
        output_dir: 输出目录
        class_mapping: 类别名称到 ID 的映射
    """
    import xml.etree.ElementTree as ET

    annotations_dir = os.path.join(voc_dir, 'Annotations')
    images_dir = os.path.join(voc_dir, 'Images')

    os.makedirs(output_dir, exist_ok=True)

    for xml_file in tqdm(os.listdir(annotations_dir)):
        if not xml_file.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        # 转换标注
        yolo_annotations = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                continue

            class_id = class_mapping[class_name]
            bbox = obj.find('bndbox')

            # VOC 格式：[xmin, ymin, xmax, ymax]
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # 转换为 YOLO 格式：[x_center, y_center, width, height] (归一化)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 保存 YOLO 格式标注
        label_file = os.path.join(output_dir, xml_file.replace('.xml', '.txt'))
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))

    print(f"VOC 格式转换完成!")


def convert_coco_to_yolo(coco_json, images_dir, output_dir, class_mapping):
    """
    将 COCO 格式转换为 YOLO 格式

    Args:
        coco_json: COCO 格式 JSON 文件路径
        images_dir: 图像目录
        output_dir: 输出目录
    """
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # 构建图像 ID 到文件名的映射
    image_map = {img['id']: img['file_name'] for img in coco_data['images']}

    # 按图像分组标注
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # 转换每个图像的标注
    for img_id, anns in tqdm(annotations_by_image.items()):
        img_file = image_map[img_id]
        img_width = coco_data['images'][coco_data['images'].index({'id': img_id})]

        # 获取图像尺寸
        img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
        img_width = img_info['width']
        img_height = img_info['height']

        yolo_annotations = []
        for ann in anns:
            category_id = ann['category_id']
            if category_id not in class_mapping:
                continue

            class_id = class_mapping[category_id]
            bbox = ann['bbox']  # [x, y, width, height]

            # 转换为 YOLO 格式
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            norm_width = bbox[2] / img_width
            norm_height = bbox[3] / img_height

            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

        # 保存
        label_file = os.path.join(output_dir, Path(img_file).stem + '.txt')
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))

    print(f"COCO 格式转换完成!")


def generate_dataset_yaml(output_path, classes, base_path="../data/urban_violations"):
    """
    生成 YOLO 格式的数据集配置文件

    Args:
        output_path: 输出 YAML 文件路径
        classes: 类别列表
        base_path: 数据集基础路径
    """
    content = f"""# 城市治理问题检测数据集配置
path: {base_path}

# 训练集/验证集/测试集路径
train: images/train
val: images/val
test: images/test

# 类别定义
names:
"""
    for i, cls in enumerate(classes):
        content += f"  {i}: {cls}\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"数据集配置文件已生成：{output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据准备工具")
    parser.add_argument("--task", choices=["split", "voc2yolo", "coco2yolo", "generate_config"],
                        help="任务类型")
    parser.add_argument("--input", help="输入目录/文件路径")
    parser.add_argument("--output", help="输出目录路径")
    parser.add_argument("--classes", nargs="+", help="类别列表")

    args = parser.parse_args()

    # 默认类别
    default_classes = [
        "illegal_parking",      # 违章停车
        "street_vendor",        # 占道经营
        "illegal_stall",        # 违规摊位
        "blocked_passage",      # 堵塞通道
        "illegal_advertisement", # 违规广告
        "garbage_dumping",      # 乱倒垃圾
        "damaged_facility"      # 设施损坏
    ]

    if args.task == "split":
        split_dataset(args.input, args.output)
    elif args.task == "voc2yolo":
        class_mapping = {cls: i for i, cls in enumerate(default_classes)}
        convert_voc_to_yolo(args.input, args.output, class_mapping)
    elif args.task == "coco2yolo":
        class_mapping = {i: i for i in range(len(default_classes))}
        convert_coco_to_yolo(args.input, args.input, args.output, class_mapping)
    elif args.task == "generate_config":
        classes = args.classes if args.classes else default_classes
        generate_dataset_yaml(args.output, classes)
