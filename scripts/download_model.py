#!/usr/bin/env python3
"""
预训练模型下载和测试脚本
无需训练即可开始测试基础检测能力
"""

from ultralytics import YOLO
import cv2


def download_and_test():
    """下载 YOLOv8 预训练模型并测试"""

    print("=" * 50)
    print("下载 YOLOv8n 预训练模型...")
    print("=" * 50)

    # 下载并加载预训练模型
    model = YOLO('yolov8n.pt')

    print("\n模型已下载！")
    print(f"模型类别：{list(model.names.values())}")

    # COCO 数据集的 80 个类别中与城市治理相关的：
    # - car (2): 汽车
    # - truck (7): 卡车
    # - bus (5): 公交车
    # - motorcycle (3): 摩托车
    # - bicycle (1): 自行车
    # - person (0): 行人
    # - traffic light (84, 不在 COCO 中)

    print("\n" + "=" * 50)
    print("预训练模型可检测的类别（COCO 80 类）：")
    print("=" * 50)

    # 打印与交通/城市相关的类别
    relevant_classes = {
        'person': '可用于检测占道经营人员',
        'bicycle': '可用于检测共享单车乱停放',
        'motorcycle': '可用于检测摩托车违规停放',
        'car': '可用于检测违章停车',
        'truck': '可用于检测货车违规停放',
        'bus': '可用于检测公交车违规停靠',
        'traffic light': '需额外训练'
    }

    for cls, desc in relevant_classes.items():
        if cls in model.names.values():
            print(f"  ✓ {cls}: {desc}")
        else:
            print(f"  ✗ {cls}: {desc} (需微调训练)")

    print("\n" + "=" * 50)
    print("下一步建议:")
    print("=" * 50)
    print("""
1. 使用预训练模型测试推理:
   python scripts/inference.py --source image --input test.jpg

2. 收集城市治理问题图像数据

3. 标注数据（使用 LabelImg 等工具）

4. 微调训练:
   python scripts/train.py

5. 部署使用
""")


if __name__ == "__main__":
    download_and_test()
