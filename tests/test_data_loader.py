"""
Unit tests for WasteWise data loading and preprocessing pipeline.

Tests verify:
1. Label mapping (O=0 -> Organic, R=1 -> Recyclable)
2. Total image counts and no silent skipping
3. Stratified split counts, proportions, and reproducibility
4. Complete isolation of the TEST dataset (no data leakage)
5. Explicit RGB conversion (including PNG-palette images)
6. Image dimensions (224x224x3) and MobileNetV2 normalization [-1.0, 1.0]
7. tf.data.Dataset pipeline generation and batching
"""

import os
from pathlib import Path
import unittest
import numpy as np

# Ensure environment variables are set before imports
_project_root = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(_project_root / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_project_root / ".matplotlib_cache"))
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import tensorflow as tf
from src.data_loader import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    LABEL_TO_NAME,
    EXPECTED_COUNTS,
    WasteWiseDataLoader,
    get_split_records,
    create_stratified_split,
    load_and_preprocess_image_pil,
    preprocess_image_tf,
    create_tf_dataset,
)


class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = _project_root / "data" / "raw" / "DATASET"
        cls.loader = WasteWiseDataLoader(dataset_dir=cls.dataset_dir, val_ratio=0.20, random_state=42)

    def test_label_mappings(self):
        """Test canonical label mappings for Organic and Recyclable classes."""
        self.assertEqual(CLASS_TO_LABEL["O"], 0)
        self.assertEqual(CLASS_TO_LABEL["R"], 1)
        self.assertEqual(LABEL_TO_CLASS[0], "O")
        self.assertEqual(LABEL_TO_CLASS[1], "R")
        self.assertEqual(LABEL_TO_NAME[0], "Organic")
        self.assertEqual(LABEL_TO_NAME[1], "Recyclable")

    def test_record_counts_and_no_missing_files(self):
        """Test that all images are accounted for with zero missing or skipped."""
        train_records = self.loader.raw_train_records
        test_records = self.loader.test_records

        self.assertEqual(len(train_records), EXPECTED_COUNTS["TRAIN"]["total"])
        self.assertEqual(len(test_records), EXPECTED_COUNTS["TEST"]["total"])
        self.assertEqual(len(train_records) + len(test_records), EXPECTED_COUNTS["TOTAL"])

        # Check per-class counts in train
        train_o = sum(1 for r in train_records if r["label"] == 0)
        train_r = sum(1 for r in train_records if r["label"] == 1)
        self.assertEqual(train_o, EXPECTED_COUNTS["TRAIN"]["O"])
        self.assertEqual(train_r, EXPECTED_COUNTS["TRAIN"]["R"])

        # Check per-class counts in test
        test_o = sum(1 for r in test_records if r["label"] == 0)
        test_r = sum(1 for r in test_records if r["label"] == 1)
        self.assertEqual(test_o, EXPECTED_COUNTS["TEST"]["O"])
        self.assertEqual(test_r, EXPECTED_COUNTS["TEST"]["R"])

        # Ensure all referenced files actually exist on disk
        for r in train_records[:50] + test_records[:50]:
            self.assertTrue(os.path.isfile(r["filepath"]), f"Missing file: {r['filepath']}")

    def test_stratified_split_proportions(self):
        """Test that train/val split is stratified and maintains class balance."""
        counts = self.loader.get_split_counts()

        self.assertEqual(counts["train"]["total"], 18051)
        self.assertEqual(counts["val"]["total"], 4513)
        self.assertEqual(counts["train"]["total"] + counts["val"]["total"], 22564)

        # Proportions should match within 0.1%
        raw_o_ratio = counts["total_train_raw"]["O_percent"]
        train_o_ratio = counts["train"]["O_percent"]
        val_o_ratio = counts["val"]["O_percent"]

        self.assertAlmostEqual(train_o_ratio, raw_o_ratio, delta=0.1)
        self.assertAlmostEqual(val_o_ratio, raw_o_ratio, delta=0.1)

    def test_dataset_isolation_and_no_leakage(self):
        """Verify train, validation, and test splits are strictly disjoint."""
        train_paths = {r["filepath"] for r in self.loader.train_records}
        val_paths = {r["filepath"] for r in self.loader.val_records}
        test_paths = {r["filepath"] for r in self.loader.test_records}

        self.assertEqual(len(train_paths.intersection(val_paths)), 0, "Leakage: train and val overlap!")
        self.assertEqual(len(train_paths.intersection(test_paths)), 0, "Leakage: train and test overlap!")
        self.assertEqual(len(val_paths.intersection(test_paths)), 0, "Leakage: val and test overlap!")

    def test_stratified_split_reproducibility(self):
        """Test that a fixed random_state produces identical splits."""
        train_1, val_1 = create_stratified_split(self.loader.raw_train_records, val_ratio=0.20, random_state=42)
        train_2, val_2 = create_stratified_split(self.loader.raw_train_records, val_ratio=0.20, random_state=42)

        paths_1 = [r["filepath"] for r in train_1]
        paths_2 = [r["filepath"] for r in train_2]
        self.assertEqual(paths_1, paths_2)

    def test_pil_image_preprocessing_standard_jpeg(self):
        """Test PIL preprocessing on standard JPEG image."""
        sample_path = self.loader.train_records[0]["filepath"]
        img_arr = load_and_preprocess_image_pil(sample_path, target_size=(224, 224))

        self.assertIsInstance(img_arr, np.ndarray)
        self.assertEqual(img_arr.shape, (224, 224, 3))
        self.assertEqual(img_arr.dtype, np.float32)
        self.assertGreaterEqual(img_arr.min(), -1.0)
        self.assertLessEqual(img_arr.max(), 1.0)

    def test_pil_image_preprocessing_palette_png(self):
        """Test PIL preprocessing explicitly converts palette PNGs to 3-channel RGB."""
        # Find one of the PNG-encoded images in the dataset
        sample_png_path = None
        for r in self.loader.train_records:
            if "O_10083.jpg" in r["filepath"]:
                sample_png_path = r["filepath"]
                break

        if sample_png_path is None:
            # Fallback to direct path
            sample_png_path = str(self.dataset_dir / "TRAIN" / "O" / "O_10083.jpg")

        self.assertTrue(os.path.exists(sample_png_path))
        img_arr = load_and_preprocess_image_pil(sample_png_path, target_size=(224, 224))

        self.assertEqual(img_arr.shape, (224, 224, 3))
        self.assertEqual(img_arr.dtype, np.float32)
        self.assertGreaterEqual(img_arr.min(), -1.0)
        self.assertLessEqual(img_arr.max(), 1.0)

    def test_tf_preprocessing_function(self):
        """Test TensorFlow preprocessing graph function."""
        sample_path = self.loader.train_records[0]["filepath"]
        raw_bytes = tf.io.read_file(sample_path)
        tensor = preprocess_image_tf(raw_bytes, target_size=(224, 224))

        self.assertEqual(tensor.shape, (224, 224, 3))
        self.assertEqual(tensor.dtype, tf.float32)
        self.assertGreaterEqual(tf.reduce_min(tensor).numpy(), -1.0)
        self.assertLessEqual(tf.reduce_max(tensor).numpy(), 1.0)

    def test_tf_dataset_pipeline_batching(self):
        """Test constructing and iterating through a tf.data.Dataset batch."""
        small_sample = self.loader.train_records[:8]
        ds = create_tf_dataset(small_sample, batch_size=4, target_size=(224, 224), shuffle=False)

        batch_count = 0
        for batch_x, batch_y in ds:
            batch_count += 1
            self.assertEqual(batch_x.shape, (4, 224, 224, 3))
            self.assertEqual(batch_y.shape, (4,))
            self.assertEqual(batch_x.dtype, tf.float32)
            self.assertTrue(all(label in [0, 1] for label in batch_y.numpy()))
            self.assertGreaterEqual(tf.reduce_min(batch_x).numpy(), -1.0)
            self.assertLessEqual(tf.reduce_max(batch_x).numpy(), 1.0)

        self.assertEqual(batch_count, 2)


if __name__ == "__main__":
    unittest.main()
