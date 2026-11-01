"""
Unit tests for WasteWise training pipeline and utilities.

Tests verify:
1. Callback creation (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger).
2. Plot generation from training history dictionaries.
3. Test set isolation verification during training pipeline setup.
"""

import os
import shutil
import tempfile
from pathlib import Path
import unittest

_project_root = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(_project_root / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_project_root / ".matplotlib_cache"))
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import tensorflow as tf
from src.train import (
    create_training_callbacks,
    plot_training_history,
)
from src.data_loader import WasteWiseDataLoader


class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_callback_creation(self):
        """Test that callbacks are configured with expected monitoring criteria."""
        ckpt_path = os.path.join(self.temp_dir, "best_model.keras")
        csv_path = os.path.join(self.temp_dir, "history.csv")

        callbacks = create_training_callbacks(
            checkpoint_path=ckpt_path,
            history_csv_path=csv_path,
            patience=3,
        )

        self.assertEqual(len(callbacks), 4)

        # Check types
        callback_types = [type(c) for c in callbacks]
        self.assertIn(tf.keras.callbacks.ModelCheckpoint, callback_types)
        self.assertIn(tf.keras.callbacks.EarlyStopping, callback_types)
        self.assertIn(tf.keras.callbacks.ReduceLROnPlateau, callback_types)
        self.assertIn(tf.keras.callbacks.CSVLogger, callback_types)

        # Check monitors
        for c in callbacks:
            if isinstance(c, tf.keras.callbacks.ModelCheckpoint):
                self.assertEqual(c.monitor, "val_loss")
                self.assertTrue(c.save_best_only)
            elif isinstance(c, tf.keras.callbacks.EarlyStopping):
                self.assertEqual(c.monitor, "val_loss")
                self.assertEqual(c.patience, 3)
                self.assertTrue(c.restore_best_weights)

    def test_plot_generation(self):
        """Test generating accuracy and loss curve images from history dict."""
        mock_history = {
            "loss": [0.65, 0.45, 0.35],
            "accuracy": [0.65, 0.80, 0.85],
            "val_loss": [0.55, 0.40, 0.32],
            "val_accuracy": [0.72, 0.82, 0.86],
        }

        output_dir = os.path.join(self.temp_dir, "outputs")
        saved_plots = plot_training_history(mock_history, output_dir=output_dir)

        self.assertIn("combined", saved_plots)
        self.assertIn("accuracy", saved_plots)
        self.assertIn("loss", saved_plots)

        for plot_key, plot_path in saved_plots.items():
            self.assertTrue(os.path.exists(plot_path), f"Plot {plot_key} not found at {plot_path}")
            self.assertGreater(os.path.getsize(plot_path), 1000, f"Plot {plot_key} is too small!")

    def test_dataset_isolation_for_training(self):
        """Verify that training and validation records have zero overlap with test records."""
        loader = WasteWiseDataLoader(val_ratio=0.20, random_state=42)
        train_paths = {r["filepath"] for r in loader.train_records}
        val_paths = {r["filepath"] for r in loader.val_records}
        test_paths = {r["filepath"] for r in loader.test_records}

        self.assertEqual(len(train_paths), 18051)
        self.assertEqual(len(val_paths), 4513)
        self.assertEqual(len(test_paths), 2513)

        self.assertEqual(len(train_paths.intersection(test_paths)), 0)
        self.assertEqual(len(val_paths.intersection(test_paths)), 0)


if __name__ == "__main__":
    unittest.main()
