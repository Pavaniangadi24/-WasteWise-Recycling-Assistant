"""
Unit tests for WasteWise evaluation pipeline and metrics calculation.

Tests verify:
1. compute_metrics with synthetic inputs (accuracy, precision, recall, F1, ROC-AUC).
2. Confusion matrix calculation (TN, FP, FN, TP) and per-class reports.
3. Plot generation for confusion matrix, ROC curve, and PR curve.
4. Model checkpoint loading verification.
"""

import os
import shutil
import tempfile
from pathlib import Path
import unittest
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(_project_root / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_project_root / ".matplotlib_cache"))
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    load_best_model,
)


class TestEvaluatePipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_metrics_synthetic(self):
        """Test metric calculation on deterministic synthetic binary predictions."""
        # 4 samples: 2 Organic (0), 2 Recyclable (1)
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = compute_metrics(y_true, y_prob, threshold=0.5)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["tn"], 2)
        self.assertEqual(metrics["confusion_matrix"]["tp"], 2)
        self.assertEqual(metrics["confusion_matrix"]["fp"], 0)
        self.assertEqual(metrics["confusion_matrix"]["fn"], 0)

    def test_compute_metrics_imperfect(self):
        """Test metric calculation on known imperfect predictions."""
        # y_true: [0, 0, 1, 1]
        # y_pred: [0, 1, 0, 1] (TN=1, FP=1, FN=1, TP=1)
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.7, 0.3, 0.8])

        metrics = compute_metrics(y_true, y_prob, threshold=0.5)

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1_score"], 0.5)
        self.assertEqual(metrics["confusion_matrix"]["tn"], 1)
        self.assertEqual(metrics["confusion_matrix"]["fp"], 1)
        self.assertEqual(metrics["confusion_matrix"]["fn"], 1)
        self.assertEqual(metrics["confusion_matrix"]["tp"], 1)
        self.assertEqual(metrics["specificity"], 0.5)

    def test_plot_generation_synthetic(self):
        """Test generation of evaluation plot images."""
        cm = np.array([[50, 5], [4, 45]])
        y_true = np.array([0] * 55 + [1] * 49)
        y_prob = np.concatenate([np.random.uniform(0, 0.4, 55), np.random.uniform(0.6, 1.0, 49)])

        cm_path = os.path.join(self.temp_dir, "cm.png")
        roc_path = os.path.join(self.temp_dir, "roc.png")
        pr_path = os.path.join(self.temp_dir, "pr.png")

        plot_confusion_matrix(cm, output_path=cm_path)
        plot_roc_curve(y_true, y_prob, auc_score=0.98, output_path=roc_path)
        plot_precision_recall_curve(y_true, y_prob, avg_precision=0.97, output_path=pr_path)

        for p in [cm_path, roc_path, pr_path]:
            self.assertTrue(os.path.exists(p))
            self.assertGreater(os.path.getsize(p), 1000)

    def test_checkpoint_exists_and_loadable(self):
        """Verify that models/best_mobilenetv2.keras exists and loads without error."""
        ckpt_path = "models/best_mobilenetv2.keras"
        self.assertTrue(os.path.isfile(ckpt_path), f"Checkpoint missing: {ckpt_path}")
        model = load_best_model(ckpt_path)
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 1))


if __name__ == "__main__":
    unittest.main()
