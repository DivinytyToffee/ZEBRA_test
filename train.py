from typing import Optional

from ultralytics import YOLO
import logging
from pathlib import Path
import torch
import sys
import time

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_log_cuda.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


def check_existing_run(run_dir: Path) -> tuple[bool, Optional[Path]]:
    """
    Checks for the existence of a previous training run and its last checkpoint.

    This function verifies if a training run directory exists and contains a valid
    'last.pt' checkpoint file. It returns a tuple indicating the presence of a checkpoint
    and the path to it, if found. Logging is used to inform about the outcome.

    Args:
        run_dir (Path): The directory path where the training run is stored.

    Returns:
        Tuple[bool, Optional[Path]]: A tuple where the first element is a boolean
        indicating if a checkpoint exists, and the second element is the path to the
        checkpoint (if it exists) or None otherwise.
    """
    if run_dir.exists():
        last_checkpoint = run_dir / 'weights' / 'last.pt'
        if last_checkpoint.exists():
            logging.info(f"Checkpoint found: {last_checkpoint}")
            return True, last_checkpoint
    logging.info("Previous training run not found")
    return False, None


def train_yolo_model(
        model_path: str,
        result_folder: str,
        runs_dir: Path,
        data_yaml: Path,
        epochs: int = 50,
        imgsz: int = 416,
        batch: int = 4,
        save_period: int = 2,
        resume: bool = True
) -> None:
    """
    Trains a YOLOv11 model with the specified parameters, using CUDA if available or falling back to CPU.

    This function initializes a YOLO model from the given model path, sets up the training device (GPU if available,
    otherwise CPU), checks for an existing training run to resume from, and trains the model using the provided dataset
    and hyperparameters. The training results and best model are saved to the specified result folder.

    Args:
        model_path (str): Path to the pre-trained YOLO model file.
        result_folder (str): Name of the folder where training results will be saved.
        data_yaml (Path): Path to the data.yaml configuration file for the dataset.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        imgsz (int, optional): Image size for training. Defaults to 416.
        batch (int, optional): Batch size for training. Defaults to 4.
        save_period (int, optional): Frequency of saving checkpoints (in epochs). Defaults to 2.
        resume (bool, optional): Whether to resume training from the last checkpoint if available. Defaults to True.
    """
    model = YOLO(model_path)
    logging.info(f"Model loaded: {model_path}")

    if torch.cuda.is_available():
        device = 'cuda:0'
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
        logging.info("Using CPU for training")

    run_dir = runs_dir / result_folder
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_status, checkpoint = check_existing_run(run_dir)

    train_params = {
        'data': str(data_yaml),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': runs_dir.name,
        'name': result_folder,
        'exist_ok': True,
        'save': True,
        'save_period': save_period,
        'verbose': True,
        'plots': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'workers': 1,
        'pretrained': True,
        'amp': False,  # Disable AMP due to nms issues
    }

    if resume or resume_status:
        if checkpoint:
            model = YOLO(checkpoint)
            train_params['resume'] = True
            logging.info("Resuming training from checkpoint")
        else:
            logging.warning("Starting training from scratch")

    start_time = time.time()
    logging.info(f"Training started with parameters: {train_params}")
    results = model.train(**train_params)
    end_time = time.time()

    logging.info(f"Training completed in {int((end_time - start_time) / 60)} minutes. Results: {results}")
    logging.info(f"Best model saved at: {run_dir / 'weights/best.pt'}")
