"""
Evaluation and Benchmarking Module for WasteWise Recycling Assistant.

Evaluates the saved best MobileNetV2 checkpoint (`models/best_mobilenetv2.keras`)
against the previously isolated test dataset (2,513 images: 1,401 Organic, 1,112 Recyclable).
Generates confusion matrices, ROC curves, PR curves, and comprehensive evaluation reports.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Configure environment variables to avoid sandbox / permission conflicts
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

os.environ.setdefault("KERAS_HOME", str(_project_root / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_project_root / ".matplotlib_cache"))
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import tensorflow as tf

from src.data_loader import get_split_records, create_tf_dataset, LABEL_TO_NAME


DEFAULT_MODEL_PATH: str = "models/best_mobilenetv2.keras"
DEFAULT_DATASET_DIR: str = "data/raw/DATASET"
DEFAULT_OUTPUT_DIR: str = "outputs"
DEFAULT_REPORTS_DIR: str = "reports"
DEFAULT_BATCH_SIZE: int = 32


def load_best_model(model_path: str = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """
    Load the saved best model checkpoint.

    Args:
        model_path: Path to the .keras model checkpoint.

    Returns:
        tf.keras.Model: Loaded Keras model instance.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"Loading best model checkpoint from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute comprehensive binary classification evaluation metrics.

    Args:
        y_true: Ground truth binary labels (0 for Organic, 1 for Recyclable).
        y_prob: Predicted probabilities for class 1 (Recyclable) in [0.0, 1.0].
        threshold: Decision threshold for discrete classification (default 0.5).

    Returns:
        Dictionary of computed metrics and classification reports.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Standard metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Per-class metrics
    class_report = classification_report(
        y_true,
        y_pred,
        target_names=["Organic", "Recyclable"],
        output_dict=True,
        zero_division=0,
    )

    # Specificity (True Negative Rate) for Organic class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": threshold,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "average_precision": float(avg_prec),
        "specificity": float(specificity),
        "confusion_matrix": {
            "tn": int(tn),  # True Organic
            "fp": int(fp),  # False Recyclable (Organic misclassified as Recyclable)
            "fn": int(fn),  # False Organic (Recyclable misclassified as Organic)
            "tp": int(tp),  # True Recyclable
            "matrix": cm.tolist(),
        },
        "per_class": {
            "Organic": {
                "precision": float(class_report["Organic"]["precision"]),
                "recall": float(class_report["Organic"]["recall"]),
                "f1_score": float(class_report["Organic"]["f1-score"]),
                "support": int(class_report["Organic"]["support"]),
            },
            "Recyclable": {
                "precision": float(class_report["Recyclable"]["precision"]),
                "recall": float(class_report["Recyclable"]["recall"]),
                "f1_score": float(class_report["Recyclable"]["f1-score"]),
                "support": int(class_report["Recyclable"]["support"]),
            },
        },
        "macro_avg": class_report["macro avg"],
        "weighted_avg": class_report["weighted avg"],
    }


def plot_confusion_matrix(cm: np.ndarray, output_path: str = "outputs/confusion_matrix.png") -> str:
    """
    Generate and save an annotated confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix array [[TN, FP], [FN, TP]].
        output_path: Path to save the image.

    Returns:
        Path to saved plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labels = ["Organic (0)", "Recyclable (1)"]

    plt.figure(figsize=(7, 6))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Format annotations with count and percentage
    annot = np.empty_like(cm).astype(str)
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            annot[i, j] = f"{count:,}\n({pct:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title("WasteWise MobileNetV2: Test Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("True Class", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, auc_score: float, output_path: str = "outputs/roc_curve.png") -> str:
    """
    Generate and save ROC curve plot.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probability for positive class.
        auc_score: Precomputed ROC-AUC score.
        output_path: Path to save the image.

    Returns:
        Path to saved plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"MobileNetV2 (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("WasteWise MobileNetV2: Test ROC Curve", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_precision_recall_curve(
    y_true: np.ndarray, y_prob: np.ndarray, avg_precision: float, output_path: str = "outputs/precision_recall_curve.png"
) -> str:
    """
    Generate and save Precision-Recall curve plot.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probability for positive class.
        avg_precision: Average Precision (AP) score.
        output_path: Path to save the image.

    Returns:
        Path to saved plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(recall_vals, precision_vals, color="#2ca02c", lw=2.5, label=f"Precision-Recall (AP = {avg_precision:.4f})")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("WasteWise MobileNetV2: Test Precision-Recall Curve", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def evaluate_test_set(
    model_path: str = DEFAULT_MODEL_PATH,
    dataset_dir: str = DEFAULT_DATASET_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """
    Execute full evaluation on the isolated test dataset.

    Args:
        model_path: Path to the saved best model (.keras).
        dataset_dir: Path to raw dataset containing TEST/.
        output_dir: Directory where evaluation outputs and plots are saved.
        batch_size: Evaluation batch size.

    Returns:
        Dictionary containing all evaluation metrics and generated paths.
    """
    print("=" * 70)
    print("Starting WasteWise Phase 5: Test Set Evaluation & Model Benchmarking")
    print("=" * 70)
    start_time = time.time()

    # 1. Load saved best model checkpoint
    model = load_best_model(model_path)

    # 2. Ingest ONLY the isolated test dataset
    print(f"\n[Step 1/4] Ingesting test dataset from: {dataset_dir}/TEST")
    test_records = get_split_records(dataset_dir, "TEST", strict_count_check=True)
    total_test = len(test_records)
    test_o = sum(1 for r in test_records if r["label"] == 0)
    test_r = sum(1 for r in test_records if r["label"] == 1)
    print(f"  Total test images: {total_test} (Organic: {test_o}, Recyclable: {test_r})")
    assert total_test == 2513, f"Expected 2513 test images, found {total_test}"

    # Build test tf.data dataset with shuffle=False to ensure deterministic sequence
    test_ds = create_tf_dataset(test_records, batch_size=batch_size, shuffle=False)

    # 3. Model Inference & Evaluation
    print("\n[Step 2/4] Running model inference on test dataset...")
    eval_results = model.evaluate(test_ds, return_dict=True, verbose=1)
    test_loss = float(eval_results["loss"])
    print(f"  Test Loss: {test_loss:.4f}")

    y_prob = model.predict(test_ds, verbose=1).flatten()
    y_true = np.array([r["label"] for r in test_records], dtype=int)

    # 4. Metrics Computation
    print("\n[Step 3/4] Computing comprehensive evaluation metrics...")
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    metrics["test_loss"] = test_loss
    metrics["total_samples"] = total_test
    metrics["organic_samples"] = test_o
    metrics["recyclable_samples"] = test_r

    # 5. Generate Visualizations & Outputs
    print("\n[Step 4/4] Generating visual artifacts and output files...")
    os.makedirs(output_dir, exist_ok=True)

    cm_array = np.array(metrics["confusion_matrix"]["matrix"])
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm_array, output_path=cm_path)

    roc_path = os.path.join(output_dir, "roc_curve.png")
    plot_roc_curve(y_true, y_prob, metrics["roc_auc"], output_path=roc_path)

    pr_path = os.path.join(output_dir, "precision_recall_curve.png")
    plot_precision_recall_curve(y_true, y_prob, metrics["average_precision"], output_path=pr_path)

    # Export predictions to CSV
    predictions_df = pd.DataFrame({
        "filepath": [r["filepath"] for r in test_records],
        "filename": [r["filename"] for r in test_records],
        "true_label": y_true,
        "true_class": [LABEL_TO_NAME[lbl] for lbl in y_true],
        "predicted_probability": np.round(y_prob, 4),
        "predicted_label": (y_prob >= 0.5).astype(int),
        "predicted_class": [LABEL_TO_NAME[(p >= 0.5).astype(int)] for p in y_prob],
        "is_correct": ((y_prob >= 0.5).astype(int) == y_true),
    })
    preds_csv_path = os.path.join(output_dir, "test_predictions.csv")
    predictions_df.to_csv(preds_csv_path, index=False)

    # Export summary JSON
    summary_json_path = os.path.join(output_dir, "test_evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    elapsed = time.time() - start_time
    metrics["evaluation_duration_sec"] = round(elapsed, 2)
    metrics["paths"] = {
        "confusion_matrix_plot": cm_path,
        "roc_curve_plot": roc_path,
        "precision_recall_plot": pr_path,
        "predictions_csv": preds_csv_path,
        "summary_json": summary_json_path,
        "model_checkpoint": model_path,
    }

    print("\n" + "=" * 70)
    print("Test Set Evaluation Complete")
    print(f"  Test Loss:        {test_loss:.4f}")
    print(f"  Accuracy:         {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision:        {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:           {metrics['recall'] * 100:.2f}%")
    print(f"  F1-Score:         {metrics['f1_score'] * 100:.2f}%")
    print(f"  ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"  Average Precision:{metrics['average_precision']:.4f}")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    evaluate_test_set()
