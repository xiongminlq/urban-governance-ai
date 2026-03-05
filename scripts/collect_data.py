#!/usr/bin/env python3
"""
数据采集工具
- 从摄像头捕获图像
- 自动截取视频帧
- 批量处理图像
"""

import cv2
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


class DataCollector:
    """数据采集器"""

    def __init__(self, output_dir="./collected_data"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def capture_from_camera(self, camera_id=0, count=100, interval=1.0):
        """
        从摄像头捕获图像

        Args:
            camera_id: 摄像头 ID
            count: 捕获数量
            interval: 捕获间隔 (秒)
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return

        print(f"开始捕获，目标数量：{count} 张，间隔：{interval} 秒")
        print("按 q 提前退出")

        captured = 0
        start_time = datetime.now()

        try:
            while captured < count:
                ret, frame = cap.read()
                if not ret:
                    break

                # 保存图像
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"capture_{timestamp}.jpg"
                filepath = self.images_dir / filename
                cv2.imwrite(str(filepath), frame)

                captured += 1
                print(f"\r已捕获：{captured}/{count}", end="", flush=True)

                # 显示预览
                cv2.putText(
                    frame,
                    f"Captured: {captured}/{count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.imshow("Data Collection", frame)

                # 等待间隔
                if cv2.waitKey(int(interval * 1000)) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n捕获完成！用时：{elapsed:.1f} 秒，保存位置：{self.images_dir}")

    def extract_frames_from_video(self, video_path, frame_interval=30):
        """
        从视频中提取帧

        Args:
            video_path: 视频文件路径
            frame_interval: 帧间隔（每 N 帧提取一帧）
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"无法打开视频：{video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"视频信息：FPS={fps}, 总帧数={total_frames}, 时长={duration:.1f}秒")

        extracted = 0
        frame_count = 0

        pbar = tqdm(total=total_frames // frame_interval, desc="提取帧")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"frame_{timestamp}_{extracted:04d}.jpg"
                filepath = self.images_dir / filename
                cv2.imwrite(str(filepath), frame)
                extracted += 1
                pbar.update(1)

            frame_count += 1

        cap.release()
        pbar.close()

        print(f"\n提取完成！共提取 {extracted} 帧图像")

    def batch_resize(self, input_dir, output_size=(640, 640)):
        """
        批量调整图像大小

        Args:
            input_dir: 输入目录
            output_size: 输出尺寸
        """
        input_path = Path(input_dir)
        output_dir = self.output_dir / "resized"
        output_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        images = [f for f in input_path.iterdir()
                  if f.suffix.lower() in image_extensions]

        for img_path in tqdm(images, desc="调整大小"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            resized = cv2.resize(img, output_size)
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), resized)

        print(f"完成！处理了 {len(images)} 张图像")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据采集工具")
    parser.add_argument("--task", choices=["capture", "extract", "resize"],
                        help="任务类型")
    parser.add_argument("--output", default="./collected_data",
                        help="输出目录")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头 ID")
    parser.add_argument("--count", type=int, default=100,
                        help="捕获数量")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="捕获间隔 (秒)")
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--frame-interval", type=int, default=30,
                        help="帧间隔")
    parser.add_argument("--input", help="输入目录")
    parser.add_argument("--size", type=int, nargs=2, default=[640, 640],
                        help="输出尺寸")

    args = parser.parse_args()

    collector = DataCollector(args.output)

    if args.task == "capture":
        collector.capture_from_camera(args.camera, args.count, args.interval)
    elif args.task == "extract":
        if not args.video:
            print("请指定视频文件路径 (--video)")
        else:
            collector.extract_frames_from_video(args.video, args.frame_interval)
    elif args.task == "resize":
        if not args.input:
            print("请指定输入目录 (--input)")
        else:
            collector.batch_resize(args.input, tuple(args.size))
