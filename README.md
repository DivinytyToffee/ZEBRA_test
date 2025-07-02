# ZEBRA_test Project

This repository contains a pipeline for training a YOLOv11 model to detect dishes, with automated prediction on video files.

## Prerequisites
- Python 3.10 or higher
- NVIDIA GPU (e.g., GeForce GTX 1060) with CUDA 11.8 compatible drivers

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/DivinytyToffee/ZEBRA_test.git
   cd ZEBRA_test
   ```

2. **Install dependencies**:
   - Run the setup script to create a virtual environment and install all required packages:
     ```bash
     python setup_dependencies.py
     ```
   - Activate the virtual environment:

## Running the Pipeline
1. **Execute the pipeline script**:
   ```bash
   python main.py
   ```
   - The script will:
     - Extract `frames.rar` and `training_video.rar` to their respective folders.
     - Run data augmentation.
     - Train the YOLOv11 model.
     - Pause for you to add custom videos to `training_video`.
     - Perform predictions on all videos and save results.

2. **Check results**:
   - Training logs are saved in `pipeline_log.log`.
   - Prediction logs are in `predict_log.log`.
   - Output videos with bounding boxes are in `runs/detect/predict_all`.
