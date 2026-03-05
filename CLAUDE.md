# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Urban Governance AI Detection System - A deep learning-based object detection system for identifying urban governance issues (illegal parking, street vendors, blocked passages, etc.) using YOLO models.

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare dataset (VOC/COCO to YOLO format, or split raw images)
python scripts/prepare_data.py --task voc2yolo --input /path/to/VOC --output data/urban_violations/labels
python scripts/prepare_data.py --task split --input data/raw_images --output data/urban_violations

# Train model
python scripts/train.py --config configs/train_config.yaml

# Inference (camera/image/video)
python scripts/inference.py --source camera --input 0
python scripts/inference.py --source image --input path/to/image.jpg
python scripts/inference.py --source video --input path/to/video.mp4 --save-video

# Web demo
python scripts/web_demo.py --port 7860

# Export model (ONNX/TensorRT)
python scripts/train.py --export models/outputs/.../best.pt --export-format onnx
```

## Architecture & Structure

### Core Components

- **scripts/train.py** - YOLOv8/v9/v10 training using Ultralytics framework
- **scripts/inference.py** - Real-time detection from camera/image/video sources
- **scripts/web_demo.py** - Gradio-based web interface
- **scripts/prepare_data.py** - Dataset conversion (VOC/COCO → YOLO) and splitting

### Configuration Files

- **configs/dataset_config.yaml** - Dataset paths and class definitions (7 classes)
- **configs/train_config.yaml** - Training hyperparameters (epochs, batch, lr, augmentations)

### Detection Classes (7 categories)

| ID | Class | Description |
|----|-------|-------------|
| 0 | illegal_parking | Vehicles parked in restricted zones |
| 1 | street_vendor | Street vendors occupying roadways |
| 2 | illegal_stall | Unauthorized fixed stalls |
| 3 | blocked_passage | Blocked pedestrian/fire exits |
| 4 | illegal_advertisement | Unauthorized advertisements |
| 5 | garbage_dumping | Illegal dumping |
| 6 | damaged_facility | Damaged public facilities |

### Data Flow

1. **Data Preparation**: Raw images → LabelImg annotation (YOLO format) → train/val/test split
2. **Training**: Load YOLO pretrained model → Train on custom dataset → Export best.pt
3. **Inference**: Load trained model → Detect frames → Draw boxes → Log violations to YAML

### Key Conventions

- All paths in scripts are relative to script location (e.g., `../configs/`, `../models/outputs/`)
- Dataset follows YOLO structure: `data/urban_violations/{images,labels}/{train,val,test}/`
- Violation records saved to `inference/results/violations.yaml` with timestamps and bounding boxes
- Default model output: `models/outputs/urban_governance_yolov8n/weights/best.pt`

### Dependencies

Core: `ultralytics`, `torch`, `opencv-python`, `pyyaml`
Optional: `onnxruntime`, `tensorrt`, `gradio`, `label-studio`
