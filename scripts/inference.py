#!/usr/bin/env python3
"""
实时推理脚本
- 从摄像头捕获视频流
- 实时检测城市治理问题
- 显示结果并保存违规记录
"""

import cv2
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import yaml


class UrbanGovernanceDetector:
    """城市治理问题检测器"""

    def __init__(self, model_path, conf_threshold=0.5, device=0):
        """
        初始化检测器

        Args:
            model_path: 模型路径
            conf_threshold: 置信度阈值
            device: 检测设备 (0=GPU, 'cpu'=CPU)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # 加载类别名称
        self.class_names = self.model.names

        # 颜色映射 (每个类别一个颜色)
        self.colors = {
            0: (0, 0, 255),      # 违章停车 - 红色
            1: (0, 255, 0),      # 占道经营 - 绿色
            2: (0, 255, 255),    # 违规摊位 - 青色
            3: (255, 0, 0),      # 堵塞通道 - 蓝色
            4: (255, 0, 255),    # 违规广告 - 紫色
            5: (128, 128, 0),    # 乱倒垃圾 - 蓝绿色
            6: (0, 128, 128),    # 设施损坏 - 红绿色
        }

        # 违规记录
        self.violations = []
        self.save_dir = Path("../inference/results")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def detect_frame(self, frame):
        """
        检测单帧图像

        Args:
            frame: 输入图像

        Returns:
            results: 检测结果
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False
        )
        return results[0]

    def draw_detections(self, frame, results):
        """
        在图像上绘制检测结果

        Args:
            frame: 原始图像
            results: 检测结果
        """
        annotated_frame = frame.copy()

        if results.boxes is None:
            return annotated_frame

        for box in results.boxes:
            # 获取边界框和类别信息
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # 获取类别名称和颜色
            class_name = self.class_names[class_id]
            color = self.colors.get(class_id, (255, 255, 255))

            # 绘制边界框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return annotated_frame

    def log_violation(self, frame, results, camera_id=0):
        """
        记录违规行为

        Args:
            frame: 当前帧
            results: 检测结果
            camera_id: 摄像头 ID
        """
        if results.boxes is None:
            return

        timestamp = datetime.now()
        save_path = self.save_dir / timestamp.strftime("%Y%m%d")
        save_path.mkdir(parents=True, exist_ok=True)

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.class_names[class_id]

            # 保存违规图像
            img_filename = f"{timestamp.strftime('%H%M%S')}_{class_name}_{confidence:.2f}.jpg"
            cv2.imwrite(str(save_path / img_filename), frame)

            # 记录违规信息
            violation = {
                "timestamp": timestamp.isoformat(),
                "camera_id": camera_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": box.xyxy[0].tolist(),
                "image_path": str(save_path / img_filename)
            }
            self.violations.append(violation)

        # 保存违规记录到 YAML
        violations_file = self.save_dir / "violations.yaml"
        with open(violations_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.violations, f, allow_unicode=True, default_flow_style=False)

    def run_camera(self, camera_id=0, save_video=False):
        """
        运行摄像头实时检测

        Args:
            camera_id: 摄像头 ID
            save_video: 是否保存视频
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return

        # 获取视频参数
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 视频保存设置
        video_writer = None
        if save_video:
            video_path = self.save_dir / f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            video_writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*'XVID'),
                fps,
                (width, height)
            )

        print("开始实时检测... (按 q 退出)")

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 检测
                results = self.detect_frame(frame)

                # 绘制结果
                annotated_frame = self.draw_detections(frame, results)

                # 显示 FPS
                frame_count += 1
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(
                    annotated_frame,
                    f"FPS: {current_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # 显示违规统计
                n_violations = len(results.boxes) if results.boxes else 0
                cv2.putText(
                    annotated_frame,
                    f"Violations: {n_violations}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                # 保存视频
                if video_writer:
                    video_writer.write(annotated_frame)

                # 记录违规
                if n_violations > 0:
                    self.log_violation(annotated_frame, results, camera_id)

                # 显示
                cv2.imshow("Urban Governance Detection", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()

        print(f"\n检测完成!")
        print(f"总违规记录：{len(self.violations)}")
        print(f"结果保存位置：{self.save_dir}")

    def run_image(self, image_path, save_result=True):
        """
        检测单张图像

        Args:
            image_path: 图像路径
            save_result: 是否保存结果
        """
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"无法读取图像：{image_path}")
            return

        results = self.detect_frame(frame)
        annotated_frame = self.draw_detections(frame, results)

        if save_result:
            output_path = self.save_dir / f"result_{Path(image_path).name}"
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"结果已保存：{output_path}")

        # 显示
        cv2.imshow("Result", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 记录违规
        self.log_violation(annotated_frame, results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="城市治理问题实时检测")
    parser.add_argument("--model", default="../models/outputs/urban_governance_yolov8n/weights/best.pt",
                        help="模型路径")
    parser.add_argument("--source", choices=["camera", "image", "video"], default="camera",
                        help="输入源类型")
    parser.add_argument("--input", type=int, default=0,
                        help="摄像头 ID 或图像/视频路径")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值")
    parser.add_argument("--save-video", action="store_true",
                        help="是否保存视频")

    args = parser.parse_args()

    detector = UrbanGovernanceDetector(args.model, args.conf)

    if args.source == "camera":
        detector.run_camera(args.input, args.save_video)
    elif args.source == "image":
        detector.run_image(args.input)
    elif args.source == "video":
        detector.run_camera(args.input, args.save_video)
