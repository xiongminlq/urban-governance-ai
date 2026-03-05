@echo off
chcp 65001 >nul
echo ================================================
echo 城市治理问题智能识别系统 - 快速启动
echo ================================================
echo.

:menu
echo 请选择要执行的操作:
echo.
echo 1. 安装依赖
echo 2. 下载预训练模型
echo 3. 数据采集 (从摄像头)
echo 4. 训练模型
echo 5. 实时检测 (摄像头)
echo 6. 图像检测
echo 7. Web Demo
echo 8. 批量检测
echo 0. 退出
echo.
set /p choice=请输入选项 (0-8):

if "%choice%"=="1" goto install
if "%choice%"=="2" goto download
if "%choice%"=="3" goto collect
if "%choice%"=="4" goto train
if "%choice%"=="5" goto detect_camera
if "%choice%"=="6" goto detect_image
if "%choice%"=="7" goto web_demo
if "%choice%"=="8" goto batch_detect
if "%choice%"=="0" goto end
goto menu

:install
echo.
echo 正在安装依赖...
pip install -r requirements.txt
echo 安装完成!
pause
goto menu

:download
echo.
echo 正在下载预训练模型...
python scripts/download_model.py
pause
goto menu

:collect
echo.
set /p count=输入要采集的图像数量 (默认 100):
set /p interval=输入采集间隔秒数 (默认 1.0):
if "%count%"=="" set count=100
if "%interval%"=="" set interval=1.0
python scripts/collect_data.py --task capture --count %count% --interval %interval%
pause
goto menu

:train
echo.
echo 开始训练模型...
echo 请确保已准备好数据集 (data/urban_violations/)
python scripts/train.py
pause
goto menu

:detect_camera
echo.
echo 启动摄像头实时检测...
echo 按 'q' 键退出
python scripts/inference.py --source camera --input 0
pause
goto menu

:detect_image
echo.
set /p imgpath=输入图像路径：
if "%imgpath%"=="" echo 未输入路径 & goto menu
python scripts/inference.py --source image --input %imgpath%
pause
goto menu

:web_demo
echo.
echo 启动 Web Demo...
echo 访问地址：http://localhost:7860
python scripts/web_demo.py --port 7860
pause
goto menu

:batch_detect
echo.
set /p inputdir=输入图像目录:
if "%inputdir%"=="" echo 未输入目录 & goto menu
python scripts/batch_detect.py --input %inputdir%
pause
goto menu

:end
echo 再见!
