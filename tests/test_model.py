"""
Unit tests for WasteWise MobileNetV2 Model Architecture.

Tests verify:
1. Model construction and layer configuration.
2. Input shape (None, 224, 224, 3).
3. Output shape (None, 1).
4. Sigmoid output range (probabilities strictly in [0.0, 1.0]).
5. Base MobileNetV2 is frozen (trainable=False).
6. Classification head layers are trainable.
7. Model compilation (loss, optimizer, metrics).
8. Forward pass execution with synthetic inputs.
9. Model save/load compatibility using Keras format (.keras).
"""

import os
import shutil
import tempfile
from pathlib import Path
import unittest
import numpy as np

# Ensure cache directories avoid sandbox permission conflicts
_project_root = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(_project_root / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_project_root / ".matplotlib_cache"))
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import tensorflow as tf
from src.model import (
    build_mobilenetv2_model,
    get_cached_weights_path,
    DEFAULT_INPUT_SHAPE,
    DEFAULT_LEARNING_RATE,
)


class TestMobileNetV2Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cached_weights = get_cached_weights_path()
        # For structural unit tests, use cached weights if available, else None (random init for shape checks)
        cls.weights_to_use = "imagenet" if cls.cached_weights is not None else None
        cls.model = build_mobilenetv2_model(
            input_shape=DEFAULT_INPUT_SHAPE,
            learning_rate=DEFAULT_LEARNING_RATE,
            weights=cls.weights_to_use,
            freeze_base=True,
            compile_model=True,
        )

    def test_pretrained_weights_active(self):
        """Test that the model uses actual ImageNet pretrained weights and not random initialization."""
        self.assertIsNotNone(self.cached_weights, "ImageNet weights should be cached locally!")
        self.assertEqual(self.weights_to_use, "imagenet")

        base_model = self.model.get_layer("mobilenetv2_1.00_224")
        conv1_weights = base_model.get_layer("Conv1").get_weights()[0]

        # Verify known ImageNet Conv1 statistics
        self.assertAlmostEqual(float(np.mean(conv1_weights)), -0.004990, delta=1e-3)
        self.assertAlmostEqual(float(np.std(conv1_weights)), 0.157948, delta=1e-3)

    def test_model_construction_and_name(self):
        """Test model is an instance of tf.keras.Model with correct name."""
        self.assertIsInstance(self.model, tf.keras.Model)
        self.assertEqual(self.model.name, "WasteWise_MobileNetV2")

    def test_input_shape(self):
        """Test input shape is exactly (None, 224, 224, 3)."""
        input_shape = self.model.input_shape
        self.assertEqual(input_shape, (None, 224, 224, 3))

    def test_output_shape(self):
        """Test output shape is exactly (None, 1) for binary classification."""
        output_shape = self.model.output_shape
        self.assertEqual(output_shape, (None, 1))

    def test_layer_composition(self):
        """Test the presence and sequence of required layers."""
        layer_names = [l.name for l in self.model.layers]
        self.assertIn("image_input", layer_names)
        self.assertIn("mobilenetv2_1.00_224", layer_names)
        self.assertIn("global_avg_pool", layer_names)
        self.assertIn("batch_norm", layer_names)
        self.assertIn("dropout_1", layer_names)
        self.assertIn("dense_128", layer_names)
        self.assertIn("dropout_2", layer_names)
        self.assertIn("classification_output", layer_names)

        # Output layer activation must be sigmoid
        out_layer = self.model.get_layer("classification_output")
        self.assertEqual(out_layer.activation.__name__, "sigmoid")

    def test_base_frozen_and_head_trainable(self):
        """Test MobileNetV2 base is frozen and classification head is trainable."""
        base_model = self.model.get_layer("mobilenetv2_1.00_224")
        self.assertFalse(base_model.trainable, "MobileNetV2 base should be frozen initially!")

        # Verify trainable parameter counts
        # Total parameters ~2,427,201; Base ~2,257,984 frozen; Trainable head ~166,657
        trainable_count = sum(tf.size(w).numpy() for w in self.model.trainable_weights)
        non_trainable_count = sum(tf.size(w).numpy() for w in self.model.non_trainable_weights)
        total_count = trainable_count + non_trainable_count

        self.assertEqual(total_count, 2427201)
        self.assertEqual(trainable_count, 166657)
        self.assertEqual(non_trainable_count, 2260544)

    def test_model_compilation(self):
        """Test optimizer, loss function, and metrics compilation."""
        self.assertIsNotNone(self.model.optimizer)
        self.assertIsInstance(self.model.optimizer, tf.keras.optimizers.Adam)
        self.assertAlmostEqual(self.model.optimizer.learning_rate.numpy(), DEFAULT_LEARNING_RATE)

        # Check loss is BinaryCrossentropy
        self.assertIsInstance(self.model.loss, tf.keras.losses.BinaryCrossentropy)
        self.assertFalse(self.model.loss.from_logits)

        # In Keras 3, check compiled user metrics
        dummy_x = np.zeros((2, 224, 224, 3), dtype=np.float32)
        dummy_y = np.zeros((2, 1), dtype=np.float32)
        eval_metrics = self.model.evaluate(dummy_x, dummy_y, return_dict=True, verbose=0)
        
        self.assertIn("loss", eval_metrics)
        self.assertIn("accuracy", eval_metrics)
        self.assertIn("precision", eval_metrics)
        self.assertIn("recall", eval_metrics)
        self.assertIn("auc", eval_metrics)

    def test_forward_pass_synthetic_input(self):
        """Test forward pass inference with synthetic normalized input [-1, 1]."""
        # Batch of 4 images with shape (4, 224, 224, 3) in range [-1.0, 1.0]
        rng = np.random.default_rng(42)
        dummy_inputs = rng.uniform(low=-1.0, high=1.0, size=(4, 224, 224, 3)).astype(np.float32)

        predictions = self.model(dummy_inputs, training=False).numpy()

        self.assertEqual(predictions.shape, (4, 1))
        # Sigmoid outputs must be strictly in [0.0, 1.0]
        self.assertTrue(np.all(predictions >= 0.0), f"Outputs below 0: {predictions}")
        self.assertTrue(np.all(predictions <= 1.0), f"Outputs above 1: {predictions}")

    def test_save_and_load_model_keras_format(self):
        """Test serializing and deserializing the compiled model in Keras format."""
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, "test_mobilenetv2.keras")
        try:
            self.model.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            loaded_model = tf.keras.models.load_model(save_path)
            self.assertEqual(loaded_model.input_shape, self.model.input_shape)
            self.assertEqual(loaded_model.output_shape, self.model.output_shape)

            # Test predictions match
            dummy_input = np.zeros((2, 224, 224, 3), dtype=np.float32)
            orig_pred = self.model(dummy_input, training=False).numpy()
            load_pred = loaded_model(dummy_input, training=False).numpy()
            np.testing.assert_allclose(orig_pred, load_pred, rtol=1e-5)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
