"""
WasteWise Image Inference and Preprocessing Helper Module.

Provides modular, reusable functions for validating uploaded images, preprocessing
them identically to the training pipeline, loading the saved MobileNetV2 model, and
computing calibrated prediction outputs and confidence scores.
"""

import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

# Set environment paths to ensure sandbox safety
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

os.environ.setdefault("KERAS_HOME", str(_project_root / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_project_root / ".matplotlib_cache"))
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

from src.data_loader import LABEL_TO_NAME, DEFAULT_TARGET_SIZE

DEFAULT_MODEL_PATH: str = "models/best_mobilenetv2.keras"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit


def validate_image_file(
    file_input: Union[str, Path, bytes, io.BytesIO],
    filename: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    Validate that an input file is an accessible, readable, uncorrupted image.

    Args:
        file_input: File path string/Path, raw bytes, or BytesIO buffer.
        filename: Optional filename for extension checking.

    Returns:
        Tuple of:
            - is_valid (bool): True if valid, False otherwise.
            - error_message (Optional[str]): Error description if invalid.
            - pil_image (Optional[Image.Image]): Loaded PIL Image object if valid.
    """
    # 1. Extension check if filename provided
    if filename:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file extension '{ext}'. Please upload JPG, JPEG, or PNG.", None

    # 2. Check byte size if bytes or buffer
    try:
        if isinstance(file_input, (str, Path)):
            p = Path(file_input)
            if not p.is_file():
                return False, f"File not found: {file_input}", None
            if p.stat().st_size == 0:
                return False, "File is empty (0 bytes).", None
            if p.stat().st_size > MAX_IMAGE_SIZE_BYTES:
                return False, f"File exceeds maximum allowed size ({MAX_IMAGE_SIZE_BYTES // (1024*1024)} MB).", None
            img = Image.open(p)
        elif isinstance(file_input, bytes):
            if len(file_input) == 0:
                return False, "File is empty (0 bytes).", None
            if len(file_input) > MAX_IMAGE_SIZE_BYTES:
                return False, f"File exceeds maximum allowed size ({MAX_IMAGE_SIZE_BYTES // (1024*1024)} MB).", None
            img = Image.open(io.BytesIO(file_input))
        elif hasattr(file_input, "read"):
            # BytesIO or Streamlit UploadedFile
            raw_bytes = file_input.read()
            if hasattr(file_input, "seek"):
                file_input.seek(0)
            if len(raw_bytes) == 0:
                return False, "Uploaded file is empty (0 bytes).", None
            if len(raw_bytes) > MAX_IMAGE_SIZE_BYTES:
                return False, f"Uploaded file exceeds maximum allowed size ({MAX_IMAGE_SIZE_BYTES // (1024*1024)} MB).", None
            img = Image.open(io.BytesIO(raw_bytes))
        else:
            return False, f"Unsupported input type: {type(file_input).__name__}", None

        # Verify image integrity
        img.verify()

        # Reopen after verify() as verify() can mutate internal state
        if isinstance(file_input, (str, Path)):
            img = Image.open(file_input)
        elif isinstance(file_input, bytes):
            img = Image.open(io.BytesIO(file_input))
        else:
            img = Image.open(file_input)

        # Check dimensions
        width, height = img.size
        if width <= 0 or height <= 0:
            return False, "Invalid image dimensions (0x0).", None

        return True, None, img

    except (UnidentifiedImageError, OSError, ValueError) as e:
        return False, f"Invalid or corrupted image: {str(e)}", None
    except Exception as e:
        return False, f"Unexpected error while reading image: {str(e)}", None


def preprocess_image_for_model(
    pil_img: Image.Image,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """
    Preprocess a PIL Image for MobileNetV2 inference:
    1. Explicitly converts to 3-channel RGB.
    2. Resizes to target_size (224, 224) using bilinear resampling.
    3. Normalizes pixel values from [0, 255] to [-1.0, 1.0] using x / 127.5 - 1.0.
    4. Adds a leading batch dimension, returning a tensor of shape (1, 224, 224, 3).

    Args:
        pil_img: PIL Image object.
        target_size: Tuple of (height, width) or (width, height), default (224, 224).

    Returns:
        np.ndarray: Float32 array of shape (1, 224, 224, 3) in range [-1.0, 1.0].
    """
    # 1. Convert to RGB mode (handles PNG 'P' palette, RGBA, Grayscale, etc.)
    rgb_img = pil_img.convert("RGB")

    # 2. Resize to 224x224
    resized_img = rgb_img.resize(target_size, Image.Resampling.BILINEAR)

    # 3. Convert to float32 numpy array
    arr = np.array(resized_img, dtype=np.float32)

    # 4. Scale to [-1.0, 1.0] exactly matching MobileNetV2 preprocessing
    scaled = (arr / 127.5) - 1.0

    # 5. Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    batch = np.expand_dims(scaled, axis=0)
    return batch


def calculate_prediction_and_confidence(
    sigmoid_output: float,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Map sigmoid probability output to class label and class-specific confidence.

    Mapping:
        0 = Organic
        1 = Recyclable

    Confidence calculation:
        If sigmoid output is p:
        - If p >= threshold: predicted class is Recyclable (1), confidence is p.
        - If p < threshold: predicted class is Organic (0), confidence is 1 - p.

    Args:
        sigmoid_output: Raw probability value from sigmoid activation in [0.0, 1.0].
        threshold: Classification decision boundary, default 0.5.

    Returns:
        Dict[str, Any] containing class name, label index, confidence, and raw probability.
    """
    p = float(sigmoid_output)
    # Clamp to [0.0, 1.0] against precision quirks
    p = max(0.0, min(1.0, p))

    if p >= threshold:
        label = 1
        category = LABEL_TO_NAME[1]  # "Recyclable"
        confidence = p
    else:
        label = 0
        category = LABEL_TO_NAME[0]  # "Organic"
        confidence = 1.0 - p

    return {
        "label": label,
        "category": category,
        "confidence": float(confidence),
        "confidence_percent": round(confidence * 100, 2),
        "raw_probability": p,
        "threshold": threshold,
    }


def predict_image(
    model: Any,
    pil_img: Image.Image,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> Dict[str, Any]:
    """
    Execute end-to-end inference on a PIL Image using the supplied Keras model.

    Args:
        model: Loaded Keras MobileNetV2 model.
        pil_img: PIL Image object.
        target_size: Target dimensions (default 224, 224).

    Returns:
        Dict[str, Any] containing prediction result, confidence, and inference latency.
    """
    start_time = time.perf_counter()

    # Preprocess
    input_tensor = preprocess_image_for_model(pil_img, target_size=target_size)

    # Predict
    raw_preds = model.predict(input_tensor, verbose=0)
    sigmoid_val = float(raw_preds[0][0])

    # Calculate class and confidence
    result = calculate_prediction_and_confidence(sigmoid_val)
    latency_ms = (time.perf_counter() - start_time) * 1000

    result["inference_latency_ms"] = round(latency_ms, 2)
    result["input_shape"] = list(input_tensor.shape)
    return result


def load_classifier_model(model_path: str = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """
    Load the trained MobileNetV2 classification model from disk.

    Args:
        model_path: Path to the .keras checkpoint file.

    Returns:
        tf.keras.Model: The loaded Keras model.
    """
    target = Path(model_path)
    if not target.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found at: {target.resolve()}")

    model = tf.keras.models.load_model(str(target))
    return model
