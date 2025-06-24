from pathlib import Path

import cv2
import os

ROOT_DIR = Path(os.getcwd())
FRAMES_DIR = ROOT_DIR / 'frames'


def extract_frames(video_path: Path, frame_rate: int = 1):
    """
        Extract frames from a video file at a specified frame rate and save them as images.

        Args:
            video_path (Path): Path to the input video file.
            frame_rate (int, optional): Number of frames to extract per second. Defaults to 1.

        Returns:
            Optional[int]: Number of frames extracted, or None if an error occurs.
        """
    output_dir = FRAMES_DIR / f'frames_for_{video_path.name.replace(video_path.suffix, "")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps / frame_rate)
    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            cv2.imwrite(str(output_dir / f"frame_{saved:06d}.jpg"), frame)
            saved += 1
        count += 1
    cap.release()
    print(f"Extracted {saved} frames from {video_path}")


def extract_all_test_frames():
    training_video = ROOT_DIR / 'training_video'
    videos = [f for f in training_video.iterdir() if f.is_file()]
    for video in videos:
        extract_frames(video)
