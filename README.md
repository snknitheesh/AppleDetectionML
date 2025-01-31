# Apple Detection using YOLO and Faster R-CNN

## Overview
This project implements an apple detection system using two different object detection models: **YOLO (You Only Look Once)** and **Faster R-CNN**. The models are trained on a custom dataset and used for detecting apples in images and video streams. The system is designed to process images, apply non-maximum suppression (NMS) for filtering detections, and save annotated results.

## Features
- Apple detection using **YOLOv8** and **Faster R-CNN**
- Training with a custom dataset
- Evaluation metrics: **mAP@0.5, mAP@0.5:0.95, Precision, Recall**
- **Non-Maximum Suppression (NMS)** for filtering overlapping detections
- Batch processing of test images
- Saves annotated images with detection results

## Dataset
The dataset is defined in `apple.yaml`, which contains the paths to training and validation image datasets. The dataset includes:
- **Bounding box annotations** for apples in each image.
- **RGB images** collected from orchards.

## Dependencies
Ensure you have the following dependencies installed:
```bash
pip install ultralytics opencv-python numpy torch torchvision
```

## Model Training
### Train YOLOv8 Model
```python
from ultralytics import YOLO

model = YOLO("/path/to/yolov8s.pt")  # Pretrained model
model.train(data="/path/to/apple.yaml", epochs=100, batch=32, imgsz=640)
```

### Train Faster R-CNN Model
checkout https://github.com/nicolaihaeni/MinneApple/tree/master for more information on training FRCNN and acquire data. 

## Object Detection and Annotation
The detection pipeline applies **NMS** to filter redundant detections and then annotates the detected apples in images.


## Example Results
![YOLO Detection](images/yolo_detection.jpg)
![Faster R-CNN Detection](images/frcnn_detection.jpg)

## Future Improvements
- Fine-tuning YOLO and Faster R-CNN with additional orchard datasets
- Implementing **real-time apple detection** on video feeds
- Deploying on **Jetson Nano/AGX Orin** for edge computing

## Author
This project is developed for advanced apple detection applications in agricultural automation.

