#!/usr/bin/env python3
"""
Web Demo 应用
使用 Gradio 构建城市治理问题检测的 Web 界面
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import yaml


class UrbanGovernanceDemo:
    """城市治理问题检测 Web Demo"""

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

        # 颜色映射
        self.colors = {
            0: (0, 0, 255),      # 违章停车 - 红色
            1: (0, 255, 0),      # 占道经营 - 绿色
            2: (0, 255, 255),    # 违规摊位 - 青色
            3: (255, 0, 0),      # 堵塞通道 - 蓝色
            4: (255, 0, 255),    # 违规广告 - 紫色
            5: (128, 128, 0),    # 乱倒垃圾 - 蓝绿色
            6: (0, 128, 128),    # 设施损坏 - 红绿色
        }

        self.violations_log = []
        self.save_dir = Path("../inference/demo_results")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def detect_image(self, image, conf_threshold):
        """
        检测图像

        Args:
            image: 输入图像 (numpy array)
            conf_threshold: 置信度阈值

        Returns:
            result_image: 标注后的图像
            violations: 检测结果文本
        """
        if image is None:
            return None, "请上传图像"

        # BGR to RGB for YOLO
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 检测
        results = self.model.predict(
            source=image_rgb,
            conf=conf_threshold,
            verbose=False
        )
        result = results[0]

        # 绘制结果
        result_image = result.plot()

        # 生成检测报告
        violations_text = "检测结果:\n\n"
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.class_names[class_id]

                violations_text += f"⚠️ {class_name}\n"
                violations_text += f"   置信度：{confidence:.2%}\n\n"

            # 保存违规记录
            self._log_violations(result, image_rgb)
        else:
            violations_text += "✅ 未检测到违规行为"

        # 转换回 BGR 用于 Gradio 显示
        result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        return result_image_bgr, violations_text

    def detect_video(self, video_file, conf_threshold):
        """
        检测视频

        Args:
            video_file: 视频文件路径
            conf_threshold: 置信度阈值

        Returns:
            output_video: 处理后的视频路径
            summary: 检测摘要
        """
        if video_file is None:
            return None, "请上传视频"

        cap = cv2.VideoCapture(video_file)

        # 获取视频参数
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 输出视频
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(self.save_dir / f"processed_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        violation_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测
            results = self.model.predict(
                source=frame,
                conf=conf_threshold,
                verbose=False
            )
            result = results[0]

            # 绘制结果
            annotated_frame = result.plot()

            # 统计违规
            if result.boxes is not None:
                violation_count += len(result.boxes)

            out.write(annotated_frame)
            frame_count += 1

        cap.release()
        out.release()

        summary = f"""
### 检测摘要
- 总帧数：{frame_count}
- 检测到违规次数：{violation_count}
- 平均每帧违规数：{violation_count / frame_count:.2f}
        """

        return output_path, summary

    def _log_violations(self, result, image):
        """记录违规行为"""
        if result.boxes is None:
            return

        timestamp = datetime.now()
        for box in result.boxes:
            class_id = int(box.cls[0])
            violation = {
                "timestamp": timestamp.isoformat(),
                "class_name": self.class_names[class_id],
                "confidence": float(box.conf[0])
            }
            self.violations_log.append(violation)

        # 保存日志
        log_file = self.save_dir / "violations_log.yaml"
        with open(log_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.violations_log, f, allow_unicode=True)

    def get_stats(self):
        """获取统计信息"""
        if not self.violations_log:
            return "暂无违规记录"

        # 统计各类别数量
        stats = {}
        for v in self.violations_log:
            name = v["class_name"]
            stats[name] = stats.get(name, 0) + 1

        text = "### 违规统计\n\n"
        for name, count in sorted(stats.items(), key=lambda x: -x[1]):
            text += f"- {name}: {count} 次\n"

        return text


def create_demo(model_path):
    """创建 Gradio Demo"""
    demo = UrbanGovernanceDemo(model_path)

    with gr.Blocks(title="城市治理问题智能识别系统") as app:
        gr.Markdown("""
        # 🏙️ 城市治理问题智能识别系统

        本系统基于深度学习技术，可自动识别以下城市治理问题：
        - 🚗 违章停车
        - 🛒 占道经营
        - 🏪 违规摊位
        - 🚧 堵塞通道
        - 📢 违规广告
        - 🗑️ 乱倒垃圾
        - 🔧 设施损坏
        """)

        with gr.Tab("图像检测"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="上传图像", type="numpy")
                    conf_slider_image = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                        label="置信度阈值"
                    )
                    detect_btn_image = gr.Button("开始检测", variant="primary")

                with gr.Column():
                    image_output = gr.Image(label="检测结果")
                    violations_text = gr.Textbox(label="检测报告", lines=10)

            detect_btn_image.click(
                fn=demo.detect_image,
                inputs=[image_input, conf_slider_image],
                outputs=[image_output, violations_text]
            )

        with gr.Tab("视频检测"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="上传视频")
                    conf_slider_video = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                        label="置信度阈值"
                    )
                    detect_btn_video = gr.Button("开始检测", variant="primary")

                with gr.Column():
                    video_output = gr.Video(label="处理结果")
                    video_summary = gr.Markdown(label="检测摘要")

            detect_btn_video.click(
                fn=demo.detect_video,
                inputs=[video_input, conf_slider_video],
                outputs=[video_output, video_summary]
            )

        with gr.Tab("统计分析"):
            stats_btn = gr.Button("刷新统计", variant="primary")
            stats_display = gr.Markdown(label="违规统计")

            stats_btn.click(
                fn=demo.get_stats,
                inputs=[],
                outputs=[stats_display]
            )

        gr.Markdown("""
        ---
        **提示**: 检测结果仅供参考，具体执法请以实际情况为准。
        """)

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="城市治理问题检测 Web Demo")
    parser.add_argument("--model", default="../models/outputs/urban_governance_yolov8n/weights/best.pt",
                        help="模型路径")
    parser.add_argument("--port", type=int, default=7860,
                        help="服务端口")
    parser.add_argument("--share", action="store_true",
                        help="是否创建公共链接")

    args = parser.parse_args()

    app = create_demo(args.model)
    app.launch(server_port=args.port, share=args.share)
