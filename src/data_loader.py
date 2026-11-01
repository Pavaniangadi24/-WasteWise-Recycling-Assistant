"""
Data loading and preprocessing module for WasteWise Recycling Assistant.

Handles loading, validating, stratified splitting, and preprocessing of the
WasteWise dataset (Organic vs. Recyclable) for MobileNetV2.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Configure environment variables to avoid sandbox/permission bottlenecks
_project_root = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(_project_root / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_project_root / ".matplotlib_cache"))
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Canonical Label Mappings
CLASS_TO_LABEL: Dict[str, int] = {
    "O": 0,  # Organic
    "R": 1,  # Recyclable
}

LABEL_TO_CLASS: Dict[int, str] = {
    0: "O",
    1: "R",
}

LABEL_TO_NAME: Dict[int, str] = {
    0: "Organic",
    1: "Recyclable",
}

# Standard Default Hyperparameters
DEFAULT_TARGET_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_VAL_RATIO: float = 0.20
DEFAULT_BATCH_SIZE: int = 32

# Expected Ground Truth Verification Counts
EXPECTED_COUNTS = {
    "TRAIN": {"O": 12565, "R": 9999, "total": 22564},
    "TEST": {"O": 1401, "R": 1112, "total": 2513},
    "TOTAL": 25077,
}


def load_and_preprocess_image_pil(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """
    Load an image from disk using PIL, explicitly convert it to RGB mode,
    resize to target_size, and normalize using MobileNetV2 preprocessing.

    Handles standard JPEG images, palette-based PNG images (mode 'P'),
    grayscale, and RGBA images safely.

    Args:
        image_path: Path to the image file.
        target_size: Desired (height, width) or (width, height) tuple (224, 224).

    Returns:
        np.ndarray: Float32 array of shape (height, width, 3) with values in [-1.0, 1.0].
    """
    path_str = str(image_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Image file not found: {path_str}")

    with Image.open(path_str) as img:
        # Explicitly convert to RGB (converts palette, RGBA, grayscale to 3-channel RGB)
        img_rgb = img.convert("RGB")
        # Resize using bilinear resampling
        img_resized = img_rgb.resize(target_size, Image.Resampling.BILINEAR)
        # Convert to numpy array of float32 in range [0, 255]
        arr = np.array(img_resized, dtype=np.float32)

    # Apply MobileNetV2 preprocessing: scales [0, 255] -> [-1.0, 1.0]
    normalized = preprocess_input(arr)
    return normalized


def preprocess_image_tf(
    image_raw: tf.Tensor,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> tf.Tensor:
    """
    TensorFlow graph-compatible preprocessing function. Decodes image bytes to 3-channel
    RGB tensor, resizes to target_size, and scales pixels to [-1.0, 1.0].

    Args:
        image_raw: String tensor of raw encoded image file bytes.
        target_size: Target (height, width) dimensions.

    Returns:
        tf.Tensor: Float32 tensor of shape (height, width, 3) in range [-1.0, 1.0].
    """
    # decode_image handles JPEG, PNG, GIF, BMP and normalizes channels to 3
    img = tf.image.decode_image(image_raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, target_size)
    img = preprocess_input(img)
    img = tf.ensure_shape(img, [target_size[0], target_size[1], 3])
    return img


def get_split_records(
    dataset_dir: Union[str, Path],
    split_name: str,
    strict_count_check: bool = True,
) -> List[Dict[str, Union[str, int]]]:
    """
    Scan the verified dataset folder for a given split ('TRAIN' or 'TEST'),
    mapping each file to its label and verifying that no files are missing
    or silently skipped.

    Args:
        dataset_dir: Root dataset directory (e.g., 'data/raw/DATASET').
        split_name: Split folder name ('TRAIN' or 'TEST').
        strict_count_check: If True, asserts exact count matches expected dataset counts.

    Returns:
        List[Dict]: List of dictionaries with keys:
            - 'filepath': Absolute path string to image.
            - 'filename': Base filename.
            - 'class_folder': Folder name ('O' or 'R').
            - 'label': Integer label (0 for Organic, 1 for Recyclable).
            - 'class_name': Human-readable string ('Organic' or 'Recyclable').
            - 'split': Split name.
    """
    split_dir = Path(dataset_dir) / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    records: List[Dict[str, Union[str, int]]] = []

    for class_folder, label_idx in CLASS_TO_LABEL.items():
        class_dir = split_dir / class_folder
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        entries = sorted([f for f in class_dir.iterdir() if f.is_file()])
        expected = EXPECTED_COUNTS.get(split_name, {}).get(class_folder)

        if strict_count_check and expected is not None:
            if len(entries) != expected:
                raise ValueError(
                    f"Count mismatch in {split_name}/{class_folder}: found {len(entries)}, expected {expected}"
                )

        for entry in entries:
            records.append({
                "filepath": str(entry.resolve()),
                "filename": entry.name,
                "class_folder": class_folder,
                "label": label_idx,
                "class_name": LABEL_TO_NAME[label_idx],
                "split": split_name,
            })

    total_found = len(records)
    expected_total = EXPECTED_COUNTS.get(split_name, {}).get("total")
    if strict_count_check and expected_total is not None and total_found != expected_total:
        raise ValueError(
            f"Total count mismatch in {split_name}: found {total_found}, expected {expected_total}"
        )

    return records


def create_stratified_split(
    train_records: List[Dict[str, Union[str, int]]],
    val_ratio: float = DEFAULT_VAL_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[List[Dict[str, Union[str, int]]], List[Dict[str, Union[str, int]]]]:
    """
    Perform a stratified split of the TRAIN records into training and validation sets.

    Ensures:
    1. Stratification preserves class balance across train and val splits.
    2. Fixed random seed ensures exact reproducibility.
    3. No images are missing or duplicated between splits.

    Args:
        train_records: List of record dicts from TRAIN split.
        val_ratio: Proportion of TRAIN records allocated to validation (default 0.20).
        random_state: Random state for reproducibility (default 42).

    Returns:
        Tuple of (train_subset, val_subset).
    """
    if not train_records:
        raise ValueError("train_records cannot be empty")

    labels = [r["label"] for r in train_records]

    train_indices, val_indices = train_test_split(
        range(len(train_records)),
        test_size=val_ratio,
        random_state=random_state,
        stratify=labels,
    )

    train_subset = [dict(train_records[i], split="train") for i in train_indices]
    val_subset = [dict(train_records[i], split="val") for i in val_indices]

    # Verify no overlap
    train_paths = {r["filepath"] for r in train_subset}
    val_paths = {r["filepath"] for r in val_subset}
    overlap = train_paths.intersection(val_paths)
    if overlap:
        raise RuntimeError(f"Data leakage detected! {len(overlap)} images overlap between train and val.")

    # Verify complete coverage
    if len(train_subset) + len(val_subset) != len(train_records):
        raise RuntimeError("Image count mismatch after stratified split.")

    return train_subset, val_subset


def create_tf_dataset(
    records: List[Dict[str, Union[str, int]]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    shuffle: bool = False,
    seed: int = DEFAULT_RANDOM_STATE,
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset pipeline from record dictionaries.

    Applies:
    - tf.io.read_file
    - decode_image with explicit 3-channel RGB handling
    - resize to target_size
    - MobileNetV2 normalization [-1.0, 1.0]
    - optional data augmentation
    - batching and prefetching for optimal GPU/CPU throughput

    Args:
        records: List of record dicts containing 'filepath' and 'label'.
        batch_size: Batch size.
        target_size: Target image dimensions (height, width).
        shuffle: Whether to shuffle the dataset.
        seed: Random seed for shuffle buffer.
        augment: Whether to apply data augmentation (random flip, slight rotation).

    Returns:
        tf.data.Dataset yielding (images, labels).
    """
    filepaths = [r["filepath"] for r in records]
    labels = [r["label"] for r in records]

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(records), 10000), seed=seed, reshuffle_each_iteration=True)

    def _parse_item(filepath_tensor, label_tensor):
        img_bytes = tf.io.read_file(filepath_tensor)
        img = preprocess_image_tf(img_bytes, target_size=target_size)

        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.1)

        label = tf.cast(label_tensor, tf.int32)
        return img, label

    dataset = dataset.map(_parse_item, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


class WasteWiseDataLoader:
    """
    Comprehensive dataset loader for WasteWise Recycling Assistant.

    Enforces:
    - Loading only from verified dataset directory (`data/raw/DATASET`).
    - Complete separation of TEST dataset.
    - Stratified splitting of TRAIN into train/validation subsets.
    - Explicit RGB conversion and MobileNetV2 preprocessing.
    - Exact verification of image counts.
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/raw/DATASET",
        val_ratio: float = DEFAULT_VAL_RATIO,
        random_state: int = DEFAULT_RANDOM_STATE,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        strict_count_check: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.target_size = target_size
        self.batch_size = batch_size
        self.strict_count_check = strict_count_check

        # Storage for records
        self.raw_train_records: List[Dict[str, Union[str, int]]] = []
        self.train_records: List[Dict[str, Union[str, int]]] = []
        self.val_records: List[Dict[str, Union[str, int]]] = []
        self.test_records: List[Dict[str, Union[str, int]]] = []

        self._load_and_partition()

    def _load_and_partition(self) -> None:
        """Load records and perform stratified partitioning."""
        # 1. Load TRAIN records
        self.raw_train_records = get_split_records(
            self.dataset_dir, "TRAIN", strict_count_check=self.strict_count_check
        )

        # 2. Stratified split into train and validation
        self.train_records, self.val_records = create_stratified_split(
            self.raw_train_records,
            val_ratio=self.val_ratio,
            random_state=self.random_state,
        )

        # 3. Load TEST records (completely separate)
        self.test_records = get_split_records(
            self.dataset_dir, "TEST", strict_count_check=self.strict_count_check
        )

        # 4. Strict leakage and disjointness verification
        self._verify_isolation()

    def _verify_isolation(self) -> None:
        """Verify absolute isolation between train, validation, and test sets."""
        train_set = {r["filepath"] for r in self.train_records}
        val_set = {r["filepath"] for r in self.val_records}
        test_set = {r["filepath"] for r in self.test_records}

        if train_set.intersection(val_set):
            raise RuntimeError("Leakage: overlap detected between train and val records.")
        if train_set.intersection(test_set):
            raise RuntimeError("Leakage: overlap detected between train and test records.")
        if val_set.intersection(test_set):
            raise RuntimeError("Leakage: overlap detected between val and test records.")

    def get_split_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Compute exact counts and class breakdown for each split.

        Returns:
            Dictionary with breakdown for train, val, test, and total.
        """
        def _breakdown(records):
            o_count = sum(1 for r in records if r["label"] == 0)
            r_count = sum(1 for r in records if r["label"] == 1)
            return {
                "total": len(records),
                "O (Organic)": o_count,
                "R (Recyclable)": r_count,
                "O_percent": round(100.0 * o_count / len(records), 2) if records else 0.0,
                "R_percent": round(100.0 * r_count / len(records), 2) if records else 0.0,
            }

        return {
            "train": _breakdown(self.train_records),
            "val": _breakdown(self.val_records),
            "test": _breakdown(self.test_records),
            "total_train_raw": _breakdown(self.raw_train_records),
            "grand_total": len(self.train_records) + len(self.val_records) + len(self.test_records),
        }

    def get_tf_datasets(
        self,
        augment_train: bool = False,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Construct tf.data.Dataset instances for train, val, and test splits.

        Args:
            augment_train: If True, applies data augmentation to the train pipeline.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        train_ds = create_tf_dataset(
            self.train_records,
            batch_size=self.batch_size,
            target_size=self.target_size,
            shuffle=True,
            seed=self.random_state,
            augment=augment_train,
        )

        val_ds = create_tf_dataset(
            self.val_records,
            batch_size=self.batch_size,
            target_size=self.target_size,
            shuffle=False,
        )

        test_ds = create_tf_dataset(
            self.test_records,
            batch_size=self.batch_size,
            target_size=self.target_size,
            shuffle=False,
        )

        return train_ds, val_ds, test_ds


if __name__ == "__main__":
    print("Testing WasteWiseDataLoader initialization...")
    loader = WasteWiseDataLoader()
    counts = loader.get_split_counts()
    print("\nDataset Partitioning Summary:")
    for split_key, stats in counts.items():
        print(f"  {split_key}: {stats}")
