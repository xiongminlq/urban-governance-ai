#!/usr/bin/env python3
"""
批量推理脚本
处理目录中的所有图像并生成报告
"""

import cv2
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from tqdm import tqdm


class BatchDetector:
    """批量检测器"""

    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names
        self.results_log = []

    def detect_directory(self, input_dir, output_dir=None, save_images=True):
        """
        批量检测目录中的图像

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            save_images: 是否保存标注图像
        """
        input_path = Path(input_dir)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path / "results"
            output_path.mkdir(parents=True, exist_ok=True)

        # 获取所有图像
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        images = [f for f in input_path.iterdir()
                  if f.suffix.lower() in image_extensions and 'result' not in f.name]

        if not images:
            print(f"在 {input_dir} 中未找到图像文件")
            return

        print(f"找到 {len(images)} 张图像，开始检测...")

        stats = {}
        pbar = tqdm(images, desc="检测中")

        for img_path in pbar:
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 检测
            results = self.model.predict(
                source=img,
                conf=self.conf_threshold,
                verbose=False
            )
            result = results[0]

            # 统计
            img_stats = {
                "file": img_path.name,
                "detections": []
            }

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.class_names[class_id]

                    img_stats["detections"].append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": box.xyxy[0].tolist()
                    })

                    # 更新统计
                    stats[class_name] = stats.get(class_name, 0) + 1

            self.results_log.append(img_stats)

            # 保存标注图像
            if save_images:
                annotated = result.plot()
                output_img = output_path / f"result_{img_path.name}"
                cv2.imwrite(str(output_img), annotated)

            pbar.set_postfix({"detected": len(img_stats["detections"])})

        # 保存 JSON 报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(images),
            "statistics": stats,
            "results": self.results_log
        }

        report_path = output_path / "detection_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印摘要
        print("\n" + "=" * 50)
        print("批量检测完成!")
        print("=" * 50)
        print(f"处理图像：{len(images)} 张")
        print(f"结果保存：{output_path}")
        print(f"检测报告：{report_path}")

        if stats:
            print("\n检测统计:")
            for cls, count in sorted(stats.items(), key=lambda x: -x[1]):
                print(f"  {cls}: {count} 次")

        return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量检测工具")
    parser.add_argument("--model", default="../models/outputs/urban_governance_yolov8n/weights/best.pt",
                        help="模型路径")
    parser.add_argument("--input", required=True,
                        help="输入图像目录")
    parser.add_argument("--output", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存标注图像")

    args = parser.parse_args()

    detector = BatchDetector(args.model, args.conf)
    detector.detect_directory(args.input, args.output, not args.no_save)
