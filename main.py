import os
import rarfile
import logging
from pathlib import Path

from augment_data import augment_data
from predict_video import predict_all_videos
from train import train_yolo_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_log.log'),
        logging.StreamHandler()
    ]
)


class_id = {
    0: 'Tea',
    1: 'greek_salat',
    2: 'ribs',
    3: 'pita',
    4: 'salat_2',
    5: 'borsch',
    6: 'pumpkin_soup',
    7: 'vodka',
    8: 'pickled_onions',
}

ROOT_DIR = Path(os.getcwd())
RUNS_DIR = ROOT_DIR / 'runs'


def extract_archives() -> None:
    """Extracts archives frames.rar and training_video.rar to their respective folders."""
    archives = {
        "frames.rar": ROOT_DIR / "frames",
        "training_video.rar": ROOT_DIR / "training_video"
    }
    for archive_name, target_dir in archives.items():
        archive_path = ROOT_DIR / archive_name
        if archive_path.exists():
            logging.info(f"Extracting {archive_name} to {target_dir}")
            with rarfile.RarFile(archive_path, 'r') as rar_ref:
                rar_ref.extractall(target_dir)
            logging.info(f"Extraction completed for {archive_name}")
        else:
            logging.warning(f"Archive {archive_name} not found")


def main(result_folder: str = "train_yolo_cuda_2"):
    """Main pipeline for model training and prediction."""
    extract_archives()

    logging.info("Starting data augmentation")
    augment_data(ROOT_DIR)
    logging.info("Data augmentation completed")

    logging.info("Starting model training")
    train_yolo_model(
        model_path="yolo11m.pt",
        result_folder=result_folder,
        data_yaml=ROOT_DIR / "dataset" / "data.yaml",
        epochs=50,
        imgsz=500,
        batch=4,
        resume=True,
        save_period=5
    )
    logging.info("Model training completed")

    input("Training completed. Please add your own videos to the 'training_video' folder and press Enter to continue...")

    logging.info("Starting video prediction")

    model_path = ROOT_DIR / "runs" / result_folder / "weights" / "best.pt"
    video_dir = ROOT_DIR / "training_video"

    predict_all_videos(model_path, video_dir, RUNS_DIR)
    logging.info("Video prediction completed")


if __name__ == "__main__":
    main()
