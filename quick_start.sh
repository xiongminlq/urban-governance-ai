#!/bin/bash
# 城市治理问题智能识别系统 - 快速启动脚本

echo "================================================"
echo "城市治理问题智能识别系统 - 快速启动"
echo "================================================"
echo ""

PS3="请选择要执行的操作 (1-9): "
options=(
    "安装依赖"
    "下载预训练模型"
    "数据采集 (从摄像头)"
    "训练模型"
    "实时检测 (摄像头)"
    "图像检测"
    "Web Demo"
    "批量检测"
    "退出"
)

select opt in "${options[@]}"
do
    case $opt in
        "安装依赖")
            echo "正在安装依赖..."
            pip install -r requirements.txt
            echo "安装完成!"
            ;;
        "下载预训练模型")
            echo "正在下载预训练模型..."
            python scripts/download_model.py
            ;;
        "数据采集 (从摄像头)")
            read -p "输入要采集的图像数量 (默认 100): " count
            read -p "输入采集间隔秒数 (默认 1.0): " interval
            count=${count:-100}
            interval=${interval:-1.0}
            python scripts/collect_data.py --task capture --count $count --interval $interval
            ;;
        "训练模型")
            echo "开始训练模型..."
            echo "请确保已准备好数据集 (data/urban_violations/)"
            python scripts/train.py
            ;;
        "实时检测 (摄像头)")
            echo "启动摄像头实时检测..."
            echo "按 'q' 键退出"
            python scripts/inference.py --source camera --input 0
            ;;
        "图像检测")
            read -p "输入图像路径: " imgpath
            python scripts/inference.py --source image --input "$imgpath"
            ;;
        "Web Demo")
            echo "启动 Web Demo..."
            echo "访问地址：http://localhost:7860"
            python scripts/web_demo.py --port 7860
            ;;
        "批量检测")
            read -p "输入图像目录：" inputdir
            python scripts/batch_detect.py --input "$inputdir"
            ;;
        "退出")
            echo "再见!"
            break
            ;;
        *) echo "无效选项";;
    esac
done
