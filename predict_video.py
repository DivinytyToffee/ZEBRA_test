import logging
from pathlib import Path
from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predict_log.log'),
        logging.StreamHandler()
    ]
)


def predict_all_videos(model_path: Path, video_dir: Path, runs_dir: Path) -> None:
    """
    Applies YOLO model predictions to all video files in the specified directory.

    This function loads a pre-trained YOLO model and processes each video file in the
    input directory, saving the output videos with bounding boxes to the output directory.

    Args:
        model_path (Path): Path to the pre-trained YOLO model file (e.g., best.pt).
        video_dir (Path): Directory containing input video files.
        runs_dir (Path): Directory containing full project

    Returns:
        None

    Raises:
        FileNotFoundError: If the model or video directory is not found.
    """
    if not model_path.exists():
        logging.error(f"Model not found at: {model_path}")
        raise FileNotFoundError(f"Model {model_path} not found")
    if not video_dir.exists():
        logging.error(f"Video directory not found at: {video_dir}")
        raise FileNotFoundError(f"Video directory {video_dir} not found")

    model = YOLO(model_path)
    logging.info(f"Model loaded successfully: {model_path}")

    video_extensions = ('.mp4', '.avi', '.mov')

    for video_file in video_dir.glob('*'):
        if video_file.is_file() and video_file.suffix.lower() in video_extensions:
            logging.info(f"Processing video: {video_file.name}")
            results = model.predict(
                source=video_file,
                save=True,
                save_txt=False,
                project=runs_dir.name,
                name=f"predict_{video_file.stem}",
                exist_ok=True,
                stream=True
            )

            for result in results:
                boxes = result.boxes
                logging.debug(f"Processed frame with {len(boxes)} detections")
            logging.info(f"Prediction completed for: {video_file.name}")
