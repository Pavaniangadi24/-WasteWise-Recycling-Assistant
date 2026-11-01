"""
MobileNetV2 Model Architecture for WasteWise Recycling Assistant.

Binary classification model to classify images as:
- Organic (label 0)
- Recyclable (label 1)

Architecture:
- Base: MobileNetV2 (include_top=False, input_shape=(224, 224, 3), weights='imagenet')
- Head: GlobalAveragePooling2D -> BatchNormalization -> Dropout(0.3) ->
        Dense(128, relu) -> Dropout(0.2) -> Dense(1, sigmoid)

Preprocessing Note:
Input images are assumed to be already preprocessed to [-1.0, 1.0] by `src/data_loader.py`.
No additional Rescaling or preprocess_input layers are added here to prevent double normalization.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

# Ensure cache directories avoid sandbox permission conflicts
_project_root = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(_project_root / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_project_root / ".matplotlib_cache"))
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import tensorflow as tf
from tensorflow.keras import layers, models


# Constants & Default Configurations
DEFAULT_INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)
DEFAULT_LEARNING_RATE: float = 0.001
MOBILENETV2_WEIGHTS_FILENAME: str = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
MOBILENETV2_WEIGHTS_URL: str = (
    "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/"
    "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
)
APPROX_WEIGHTS_SIZE_MB: float = 9.4


def get_cached_weights_path() -> Optional[Path]:
    """
    Check if the ImageNet pretrained weights file for MobileNetV2 is cached locally.

    Searches:
    1. Project local `.keras/models/`
    2. User home `~/.keras/models/`
    3. Project `models/` directory

    Returns:
        Path if found, else None.
    """
    search_paths = [
        Path(os.environ["KERAS_HOME"]) / "models" / MOBILENETV2_WEIGHTS_FILENAME,
        Path.home() / ".keras" / "models" / MOBILENETV2_WEIGHTS_FILENAME,
        _project_root / "models" / MOBILENETV2_WEIGHTS_FILENAME,
    ]
    for p in search_paths:
        try:
            if p.is_file() and p.stat().st_size > 0:
                return p
        except (PermissionError, OSError):
            continue
    return None


def build_mobilenetv2_model(
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weights: Optional[str] = "imagenet",
    freeze_base: bool = True,
    compile_model: bool = True,
) -> tf.keras.Model:
    """
    Construct, configure, and optionally compile the WasteWise MobileNetV2 binary classifier.

    Args:
        input_shape: Input image tensor dimensions (height, width, channels).
                     Defaults to (224, 224, 3).
        learning_rate: Initial learning rate for the Adam optimizer (default: 0.001).
        weights: Pretrained weights to load ('imagenet', None, or path to local .h5 weights file).
        freeze_base: Whether to freeze base MobileNetV2 layers for transfer learning feature extraction.
        compile_model: Whether to compile the model with Adam and BinaryCrossentropy.

    Returns:
        tf.keras.Model: Configured MobileNetV2 binary classification model.
    """
    # Check weights availability if 'imagenet' is requested
    base_weights = weights
    if weights == "imagenet":
        cached_path = get_cached_weights_path()
        if cached_path is not None:
            base_weights = str(cached_path)
        else:
            # Check if Keras can resolve 'imagenet' directly or if offline error occurs
            # We preserve weights="imagenet" so Keras can load if cached in internal store,
            # but if it fails, caller receives clear diagnostic.
            base_weights = "imagenet"

    # 1. Base MobileNetV2 Model
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=base_weights,
        )
    except Exception as e:
        if weights == "imagenet" and ("getaddrinfo failed" in str(e) or "URL fetch failure" in str(e)):
            raise RuntimeError(
                f"ImageNet pretrained weights file ({MOBILENETV2_WEIGHTS_FILENAME}, ~{APPROX_WEIGHTS_SIZE_MB} MB) "
                f"is not cached locally. User permission is required to download weights from:\n"
                f"{MOBILENETV2_WEIGHTS_URL}\n"
                f"Original error: {e}"
            ) from e
        raise e

    # 2. Freeze Base Model layers for initial feature extraction
    base_model.trainable = not freeze_base

    # 3. Define Model Inputs (No redundant Rescaling/preprocess_input layer)
    inputs = layers.Input(shape=input_shape, name="image_input")

    # 4. Forward through MobileNetV2 base (in inference mode when frozen)
    x = base_model(inputs, training=False if freeze_base else None)

    # 5. Custom Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="classification_output")(x)

    # 6. Construct Functional Model
    model = models.Model(inputs=inputs, outputs=outputs, name="WasteWise_MobileNetV2")

    # 7. Compile Model
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

    return model


if __name__ == "__main__":
    print("Testing build_mobilenetv2_model...")
    cached = get_cached_weights_path()
    print(f"Cached ImageNet weights path: {cached}")

    # Build model using weights=None if not yet cached, to inspect architecture
    test_weights = "imagenet" if cached is not None else None
    print(f"Building model with weights={test_weights}...")
    m = build_mobilenetv2_model(weights=test_weights)
    m.summary()
