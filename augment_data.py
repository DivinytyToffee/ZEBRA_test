import random
import shutil
from pathlib import Path
import cv2
import albumentations as A
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
NUM_AUGMENTATIONS = 5


def check_yolo_annotations(image_dir: Path, label_dir: Path) -> bool:
    """
    Check the validity of YOLO annotation files and their corresponding images.

    Args:
        image_dir (Path): Directory containing image files.
        label_dir (Path): Directory containing YOLO annotation files.

    Raises:
        ValueError: If directories are invalid or annotations have incorrect format.
    """
    check = True
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError("Image or label directory does not exist")

    image_files = {f.stem for f in image_dir.iterdir() if f.suffix.lower() in {'.jpg', '.png'}}
    label_files = {f.stem for f in label_dir.iterdir() if f.suffix == '.txt'}

    missing_images = label_files - image_files
    missing_labels = image_files - label_files
    if missing_images:
        logging.warning(f"Missing images for labels: {missing_images}")
    if missing_labels:
        logging.warning(f"Missing labels for images: {missing_labels}")

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                values = line.strip().split()
                if len(values) != 5:
                    logging.error(f"Invalid format in {label_file}, line {line_num}: {line}")
                    check = False
                    break
                try:
                    class_id, x, y, w, h = map(float, values)
                    if not (0 <= class_id <= 100):
                        logging.error(f"Invalid class_id in {label_file}, line {line_num}: {class_id}")
                        check = False
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        logging.error(f"Invalid coordinates in {label_file}, line {line_num}: {line}")
                        check = False
                except ValueError:
                    logging.error(f"Non-numeric values in {label_file}, line {line_num}: {line}")
                    check = False
    return check


def augment_image_and_bboxes(
        image_path: Path,
        label_path: Path,
        output_image_dir: Path,
        output_label_dir: Path,
        augmentations: A.Compose,
) -> None:
    """
    Apply augmentations to an image and its YOLO annotations, saving the results.

    Args:
        image_path (Path): Path to the input image.
        label_path (Path): Path to the YOLO annotation file.
        output_image_dir (Path): Directory to save augmented images.
        output_label_dir (Path): Directory to save augmented annotations.
        augmentations (A.Compose): Albumentations pipeline.
    """
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return

    bboxes = []
    class_labels = []
    if not label_path.exists():
        logging.warning(f"No label file for {image_path}, skipping")
        return

    with open(label_path, 'r') as f:
        for line in f:
            class_id, x, y, w, h = map(float, line.strip().split())
            bboxes.append([x, y, w, h])
            class_labels.append(int(class_id))

    for i in range(NUM_AUGMENTATIONS):
        aug = augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_image = aug['image']
        aug_bboxes = aug['bboxes']
        aug_labels = aug['class_labels']

        aug_image_path = output_image_dir / f"{image_path.stem}_aug{i}.jpg"
        cv2.imwrite(str(aug_image_path), aug_image)

        aug_label_path = output_label_dir / f"{image_path.stem}_aug{i}.txt"
        with open(aug_label_path, 'w') as f:
            for bbox, label in zip(aug_bboxes, aug_labels):
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    f.write(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def augment_dataset(image_dir: Path, label_dir: Path, output_image_dir: Path, output_label_dir: Path) -> None:
    """
    Augment all images and annotations in the dataset.

    Args:
        image_dir (Path): Directory with original images.
        label_dir (Path): Directory with original YOLO annotations.
        output_image_dir (Path): Directory to save augmented images.
        output_label_dir (Path): Directory to save augmented annotations.
    """
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError("Image or label directory does not exist")

    transform = A.Compose(
        [
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.1, rotate=(-15, 15), p=0.5),
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            min_visibility=0.3,
            label_fields=['class_labels']
        )
    )

    for image_path in image_dir.glob("*.jpg"):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            logging.warning(f"No label file for {image_path}, skipping")
            continue
        augment_image_and_bboxes(image_path, label_path, output_image_dir, output_label_dir, transform)


def split_dataset(
        image_dir: Path,
        label_dir: Path,
        output_dir: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
) -> None:
    """
    Split dataset into train, validation, and test sets.

    Args:
        image_dir (Path): Directory with images.
        label_dir (Path): Directory with YOLO annotations.
        output_dir (Path): Directory to save split dataset.
        train_ratio (float): Ratio of images for training.
        val_ratio (float): Ratio of images for validation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.glob("*.jpg"))
    random.shuffle(image_files)
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = [
        ('train', image_files[:train_end]),
        ('val', image_files[train_end:val_end]),
        ('test', image_files[val_end:])
    ]

    for split, files in splits:
        for image_path in files:
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy(image_path, output_dir / 'images' / split / image_path.name)
                shutil.copy(label_path, output_dir / 'labels' / split / label_path.name)


def augment_data(root_dir: Path) -> None:
    """
    Augments a dataset by applying transformations to images and their corresponding YOLO annotations.

    This function processes images and labels from specified directories, performs augmentation,
    and splits the dataset into training and validation sets. It checks the integrity of YOLO
    annotations before proceeding. Augmented data is saved to designated output directories.

    Args:
        root_dir (Path): Path to the root directory

    Raises:
        FileNotFoundError: If input directories or files are missing.
        ValueError: If annotations are incorrect or incompatible.

    Notes:
        - Requires 'check_yolo_annotations', 'augment_dataset', and 'split_dataset' functions to be defined.
        - Logging is used to track the process and report issues.
    """
    image_dir: Path = root_dir / 'frames'
    label_dir: Path = root_dir / 'data' / 'obj_train_data'
    output_image_dir: Path = root_dir / 'dataset' / 'augmented' / 'images'
    output_label_dir: Path = root_dir / 'dataset' / 'augmented' / 'labels'
    check: bool = check_yolo_annotations(image_dir, label_dir)
    if check:
        augment_dataset(
            image_dir,
            label_dir,
            output_image_dir,
            output_label_dir
        )

        split_dataset(
            output_image_dir,
            output_label_dir,
            root_dir / "dataset"
        )

        logging.info('Done!')
    else:
        logging.warning("Annotations uncorrected")
