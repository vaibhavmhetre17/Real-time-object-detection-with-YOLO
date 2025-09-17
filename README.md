
# Real-Time Object Detection using YOLOv8

This project is about detecting objects in real time with **YOLOv8**.  
The model can take input from images, videos, or even a webcam and highlight the detected objects with bounding boxes.

---

## Features
- Train the model on a custom dataset
- Run detection on images, videos, or live camera
- Save the results with bounding boxes and labels
- Check accuracy and performance using evaluation metrics

---

## Why YOLOv8?
We decided to use **YOLOv8** because:
- It gives good accuracy while being fast
- It is easy to train and test in one framework
- It has many tutorials and documentation available
- It supports transfer learning with pretrained weights  

(We are aware that **YOLOv11** is newer and gives better results, but YOLOv8 is more stable, easier to use, and well supported right now. So we selected YOLOv8 for this project.)

---

## Required Python Libraries

| Library            | Purpose                                           |
|--------------------|---------------------------------------------------|
| ultralytics        | Main YOLOv8 library                               |
| torch, torchvision | Deep learning framework (PyTorch)                 |
| opencv-python      | Image/video handling                              |
| numpy              | Array and math operations                         |
| pandas             | Working with dataset files                        |
| matplotlib         | Visualizing results and training metrics          |
| seaborn            | Extra plots (like confusion matrix)               |
| pyyaml             | Reading dataset configuration files               |
| tqdm               | Progress bar during training                      |
| scikit-learn       | Splitting dataset, evaluation (optional)          |
| scipy              | Numerical functions (optional)                    |
| tensorboard        | Monitoring training progress (optional)           |
| roboflow           | Dataset import (optional)                         |
| psutil             | Monitor CPU/GPU usage (optional)                  |



## Research & Select YOLOv8 Variant

We analyzed different YOLOv8 pre-trained models based on size, speed, accuracy, and hardware requirements:

### YOLOv8-n (Nano)
- **Size & Speed:** Extremely lightweight with the fewest parameters. Offers the fastest inference and training speed.  
- **Hardware Efficiency:** Can run efficiently on CPUs or low-end GPUs, making it suitable for edge devices or real-time applications.  
- **Accuracy:** Slightly lower than larger models but sufficient for many practical use-cases.  
- **Use-case:** Best when speed and low resource usage are more critical than maximum accuracy.

### YOLOv8-s (Small)
- Slightly larger than Nano, with better accuracy.  
- Moderate inference speed, suitable for standard GPUs.  
- Good balance between speed and accuracy for projects with mid-range hardware.

### YOLOv8-m (Medium), YOLOv8-l (Large), YOLOv8-x (Extra Large)
- Very high accuracy.  
- Requires powerful GPUs for training and inference.  
- Slower and heavier, not ideal for real-time or low-resource environments.

### Selection Justification
For this project, the focus is **real-time object detection** with limited hardware resources (CPU or low-end GPU). **YOLOv8-n** was selected because it provides **fast inference, low memory usage, and acceptable accuracy**, which aligns perfectly with the project’s speed and efficiency requirements.


## Baseline Training

To establish a baseline for future experiments, we trained **YOLOv8n** on our dataset for a small number of epochs. This helps track performance improvements over time.

### Training Details

- **Model:** YOLOv8n  
- **Epochs:** 10  
- **Batch size:** 16  
- **Image size:** 640  
- **Experiment name:** yolov8_baseline5  
- **Optimizer:** AdamW (auto)  
- **Training dataset:** 135 images (train), 55 images (validation)  

### Validation Results (Last Epoch)

| Metric       | Value      |
|--------------|------------|
| mAP@0.5      | 0.00723    |
| mAP@0.5-0.95 | 0.0057     |
| Box loss     | 1.192      |
| Cls loss     | 4.581      |
| DFL loss     | 1.241      |

### Notes

- Training completed successfully for 10 epochs.  
- Baseline performance is low due to the small dataset and CPU training.  
- Results are saved in `runs/detect/yolov8_baseline5`.  
- This baseline will be used for comparison with future experiments using more epochs, larger models, or GPU acceleration.

### Validation Results (Best Weights)
| Metric       | Value|
| -------------|------|
| Precision    | 0.56 |
| Recall       | 0.60 |
| mAP@0.5      | 0.58 |
| mAP@0.5-0.95 | 0.36 |

### Hyperparameter Tuning for YOLOv8

This task focuses on improving the performance of YOLOv8 through hyperparameter optimization.
We used Weights & Biases (W&B) Sweeps to test different parameter combinations and compare them with a baseline model.

### Features

- Train YOLOv8 baseline with default parameters
- Define and run hyperparameter sweeps using W&B
- Automatically explore learning rate, batch size, image size, and epochs
- Compare baseline vs tuned performance
- Save all results in runs/ and W&B dashboard

### Why Hyperparameter Tuning?

- Default YOLOv8 settings are not always optimal
- Parameters like learning rate and batch size can heavily impact accuracy and loss
- Manual tuning is time-consuming
- W&B Sweeps automate the process and provide systematic comparisons

### Sweep Training
  Pperformed automated sweeps  
  Parameters explored:
- **Learning rate:** 0.001 – 0.01
- **Batch size:** [16, 32, 64]
- **Image size:** [416, 512, 640]
- **Epochs:** 50

### Notes

- W&B Sweeps automated the hyperparameter search efficiently.
- Tuned configuration improved performance compared to the baseline.
- Results are reproducible and stored both locally and in W&B.

