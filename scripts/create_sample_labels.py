# YOLO 格式标注文件生成工具
# 使用此脚本可以快速创建示例标注文件用于测试

import os
from pathlib import Path


def create_sample_labels(output_dir, classes):
    """
    创建示例 YOLO 格式标注文件

    Args:
        output_dir: 输出目录
        classes: 类别列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建示例标注内容
    # 格式：class_id x_center y_center width height (归一化坐标)
    sample_data = []

    # 示例：违章停车
    sample_data.append(f"0 0.5 0.5 0.3 0.4")

    # 示例：占道经营
    sample_data.append(f"1 0.3 0.6 0.2 0.3")

    # 写入文件
    sample_file = output_path / "sample_label.txt"
    with open(sample_file, 'w') as f:
        f.write('\n'.join(sample_data))

    print(f"示例标注文件已创建：{sample_file}")
    print("\nYOLO 标注格式说明:")
    print("  每行一个目标：class_id x_center y_center width height")
    print("  坐标为归一化值 (0-1)")
    print("  x_center, y_center: 边界框中心点相对图像宽高的比例")
    print("  width, height: 边界框宽高相对图像宽高的比例")

    # 创建数据集配置文件
    config_content = f"""# 数据集配置
path: {output_path.parent}
train: images/train
val: images/val
test: images/test

names:
"""
    for i, cls in enumerate(classes):
        config_content += f"  {i}: {cls}\n"

    config_file = output_path.parent / "dataset.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)

    print(f"\n数据集配置文件已创建：{config_file}")


if __name__ == "__main__":
    classes = [
        "illegal_parking",      # 违章停车
        "street_vendor",        # 占道经营
        "illegal_stall",        # 违规摊位
        "blocked_passage",      # 堵塞通道
        "illegal_advertisement", # 违规广告
        "garbage_dumping",      # 乱倒垃圾
        "damaged_facility"      # 设施损坏
    ]

    create_sample_labels("./data/sample_labels", classes)
