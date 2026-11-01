"""
Training module for WasteWise MobileNetV2 Transfer Learning.

Trains the classification head on top of the frozen MobileNetV2 base using the
stratified 80/20 train/validation split (18,051 train, 4,513 val).
The test dataset (2,513 images) is strictly preserved and never touched.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Configure environment variables to prevent sandbox permission bottlenecks
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
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)

from src.data_loader import WasteWiseDataLoader, create_tf_dataset
from src.model import build_mobilenetv2_model


# Standard Default Training Configuration
DEFAULT_EPOCHS: int = 15
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LEARNING_RATE: float = 0.001
DEFAULT_PATIENCE: int = 3
DEFAULT_CHECKPOINT_PATH: str = "models/best_mobilenetv2.keras"
DEFAULT_OUTPUT_DIR: str = "outputs"
DEFAULT_REPORTS_DIR: str = "reports"


def create_training_callbacks(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    history_csv_path: str = "outputs/training_history.csv",
    patience: int = DEFAULT_PATIENCE,
) -> list:
    """
    Configure callbacks for model checkpointing, early stopping, learning rate reduction,
    and CSV logging.

    Args:
        checkpoint_path: Destination path for best model (.keras format).
        history_csv_path: Destination path for epoch-by-epoch CSV log.
        patience: Number of epochs to wait before early stopping.

    Returns:
        List of configured Keras callback instances.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_csv_path), exist_ok=True)

    callbacks = [
        # Save only the best model based on validation loss
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        # Early stopping to prevent overfitting and save compute
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        # Persist metrics per epoch to CSV
        CSVLogger(
            filename=history_csv_path,
            separator=",",
            append=False,
        ),
    ]
    return callbacks


def plot_training_history(history_dict: Dict[str, list], output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict[str, str]:
    """
    Generate and save publication-quality plots for training and validation metrics.

    Args:
        history_dict: Dictionary containing metric lists across epochs.
        output_dir: Directory where plot images will be saved.

    Returns:
        Dictionary mapping plot names to saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs_range = range(1, len(history_dict["loss"]) + 1)
    saved_plots = {}

    # 1. Combined Accuracy and Loss figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs_range, history_dict["accuracy"], "b-o", label="Training Accuracy", linewidth=2)
    ax1.plot(epochs_range, history_dict["val_accuracy"], "r--s", label="Validation Accuracy", linewidth=2)
    ax1.set_title("Training vs. Validation Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_xticks(list(epochs_range))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss Plot
    ax2.plot(epochs_range, history_dict["loss"], "b-o", label="Training Loss", linewidth=2)
    ax2.plot(epochs_range, history_dict["val_loss"], "r--s", label="Validation Loss", linewidth=2)
    ax2.set_title("Training vs. Validation Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Binary Crossentropy Loss", fontsize=11)
    ax2.set_xticks(list(epochs_range))
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(combined_plot_path, dpi=300)
    plt.close(fig)
    saved_plots["combined"] = combined_plot_path

    # 2. Standalone Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_dict["accuracy"], "b-o", label="Train Accuracy", linewidth=2)
    plt.plot(epochs_range, history_dict["val_accuracy"], "r--s", label="Val Accuracy", linewidth=2)
    plt.title("WasteWise MobileNetV2: Accuracy Curves", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.xticks(list(epochs_range))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    saved_plots["accuracy"] = acc_path

    # 3. Standalone Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_dict["loss"], "b-o", label="Train Loss", linewidth=2)
    plt.plot(epochs_range, history_dict["val_loss"], "r--s", label="Val Loss", linewidth=2)
    plt.title("WasteWise MobileNetV2: Loss Curves", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Binary Crossentropy Loss", fontsize=11)
    plt.xticks(list(epochs_range))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    saved_plots["loss"] = loss_path

    return saved_plots


def train_model(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    patience: int = DEFAULT_PATIENCE,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    val_ratio: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute full transfer learning training pipeline for the MobileNetV2 classification head.

    Args:
        epochs: Maximum number of epochs.
        batch_size: Training batch size.
        learning_rate: Initial Adam learning rate.
        patience: Early stopping patience.
        checkpoint_path: Path to save the best model checkpoint.
        output_dir: Output directory for plots and history logs.
        val_ratio: Proportion of TRAIN dataset allocated for validation.
        random_state: Fixed seed for data partitioning reproducibility.

    Returns:
        Dictionary containing training summary, best metrics, and paths.
    """
    print("=" * 70)
    print("Starting WasteWise Phase 4: Model Training (Classification Head)")
    print("=" * 70)
    start_time = time.time()

    # 1. Initialize data loader and retrieve train/val records
    print("\n[Step 1/5] Ingesting verified dataset and generating stratified splits...")
    loader = WasteWiseDataLoader(
        dataset_dir="data/raw/DATASET",
        val_ratio=val_ratio,
        random_state=random_state,
        batch_size=batch_size,
    )

    split_counts = loader.get_split_counts()
    print(f"  Training samples:   {split_counts['train']['total']} (Organic: {split_counts['train']['O (Organic)']}, Recyclable: {split_counts['train']['R (Recyclable)']})")
    print(f"  Validation samples: {split_counts['val']['total']} (Organic: {split_counts['val']['O (Organic)']}, Recyclable: {split_counts['val']['R (Recyclable)']})")
    print(f"  Test samples:       {split_counts['test']['total']} [STRICTLY PRESERVED & UNTOUCHED]")

    # Build tf.data pipelines only for train and validation (test is never loaded)
    train_ds = create_tf_dataset(
        loader.train_records,
        batch_size=batch_size,
        shuffle=True,
        seed=random_state,
        augment=False,
    )
    val_ds = create_tf_dataset(
        loader.val_records,
        batch_size=batch_size,
        shuffle=False,
    )

    # 2. Build MobileNetV2 model with frozen base and ImageNet weights
    print("\n[Step 2/5] Building MobileNetV2 with ImageNet weights and frozen base...")
    model = build_mobilenetv2_model(
        weights="imagenet",
        freeze_base=True,
        compile_model=True,
        learning_rate=learning_rate,
    )
    print("  Base MobileNetV2 is frozen. Trainable parameters: 166,657 / 2,427,201.")

    # 3. Setup Callbacks
    print("\n[Step 3/5] Configuring training callbacks...")
    history_csv_path = os.path.join(output_dir, "training_history.csv")
    callbacks = create_training_callbacks(
        checkpoint_path=checkpoint_path,
        history_csv_path=history_csv_path,
        patience=patience,
    )

    # 4. Train Model
    print(f"\n[Step 4/5] Training for up to {epochs} epochs (batch_size={batch_size})...")
    training_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    training_duration = time.time() - training_start

    # Convert history values to standard floats
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # 5. Extract Best Metrics & Epoch
    best_epoch_idx = int(tf.argmin(history_dict["val_loss"]).numpy())
    best_epoch = best_epoch_idx + 1
    total_epochs_completed = len(history_dict["loss"])
    early_stopped = total_epochs_completed < epochs

    results = {
        "status": "success",
        "total_training_duration_sec": round(training_duration, 2),
        "total_epochs_completed": total_epochs_completed,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "batch_size": batch_size,
        "initial_learning_rate": learning_rate,
        "patience": patience,
        "best_epoch_metrics": {
            "epoch": best_epoch,
            "train_loss": round(history_dict["loss"][best_epoch_idx], 4),
            "train_accuracy": round(history_dict["accuracy"][best_epoch_idx], 4),
            "train_precision": round(history_dict.get("precision", [0])[best_epoch_idx], 4),
            "train_recall": round(history_dict.get("recall", [0])[best_epoch_idx], 4),
            "train_auc": round(history_dict.get("auc", [0])[best_epoch_idx], 4),
            "val_loss": round(history_dict["val_loss"][best_epoch_idx], 4),
            "val_accuracy": round(history_dict["val_accuracy"][best_epoch_idx], 4),
            "val_precision": round(history_dict.get("val_precision", [0])[best_epoch_idx], 4),
            "val_recall": round(history_dict.get("val_recall", [0])[best_epoch_idx], 4),
            "val_auc": round(history_dict.get("val_auc", [0])[best_epoch_idx], 4),
        },
        "final_epoch_metrics": {
            "epoch": total_epochs_completed,
            "train_loss": round(history_dict["loss"][-1], 4),
            "train_accuracy": round(history_dict["accuracy"][-1], 4),
            "val_loss": round(history_dict["val_loss"][-1], 4),
            "val_accuracy": round(history_dict["val_accuracy"][-1], 4),
        },
        "history": history_dict,
        "checkpoint_path": checkpoint_path,
        "history_csv_path": history_csv_path,
    }

    # Save JSON history
    history_json_path = os.path.join(output_dir, "training_summary.json")
    with open(history_json_path, "w") as f:
        json.dump(results, f, indent=2)
    results["summary_json_path"] = history_json_path

    # Generate plots
    print("\n[Step 5/5] Generating training curves and plots...")
    saved_plots = plot_training_history(history_dict, output_dir=output_dir)
    results["plots"] = saved_plots

    total_duration = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Training Complete in {total_duration:.2f}s ({total_duration / 60:.2f} min)")
    print(f"Epochs Completed: {total_epochs_completed} / {epochs}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {results['best_epoch_metrics']['val_loss']:.4f}")
    print(f"Best Val Accuracy: {results['best_epoch_metrics']['val_accuracy'] * 100:.2f}%")
    print(f"Best Val AUC: {results['best_epoch_metrics']['val_auc']:.4f}")
    print(f"Saved Best Model: {checkpoint_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    train_model()
