"""
Prediction module for brain tumor classification
Handles model loading, image preprocessing, and inference
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "notebooks" / "exploration" / "brain_tumor_cnn_improved"

LABEL_MAP = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary"
}

TARGET_SIZE = (192, 192)
CLIP_PCT = (1, 99)

_model = None

def load_model():
    """Load and cache the trained model"""
    global _model
    if _model is None:
        loaded = tf.saved_model.load(str(MODEL_PATH))
        _model = loaded.signatures['serving_default']
    return _model

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image to match training pipeline
    Steps: grayscale conversion, resize, percentile clipping, normalization
    """
    # Convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        img = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        img = image

    img = img.astype(np.float32)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # Percentile clipping for robust normalization
    lo, hi = np.percentile(img, CLIP_PCT)
    img = np.clip(img, lo, hi)

    # Normalize to [0, 1] then convert to uint8
    denom = (hi - lo) if hi > lo else 1.0
    img = (img - lo) / denom
    img = (img * 255.0).round().astype(np.uint8)

    # Convert back to float32 for model input
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=-1)

    return img

def predict_single(image: np.ndarray) -> dict:
    """Predict tumor type for a single MRI image"""
    model = load_model()
    processed = preprocess_image(image)
    batch = np.expand_dims(processed, axis=0)

    output = model(input_1=tf.constant(batch, dtype=tf.float32))
    predictions = output['output_0'].numpy()

    class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx])
    label = LABEL_MAP[class_idx]

    probabilities = {LABEL_MAP[i]: float(predictions[0][i]) for i in range(len(LABEL_MAP))}

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities
    }

def predict_batch(images: list) -> list:
    """Predict tumor types for multiple MRI images"""
    model = load_model()
    processed = np.array([preprocess_image(img) for img in images])

    output = model(input_1=tf.constant(processed, dtype=tf.float32))
    predictions = output['output_0'].numpy()

    results = []
    for i, pred in enumerate(predictions):
        class_idx = int(np.argmax(pred))
        confidence = float(pred[class_idx])
        label = LABEL_MAP[class_idx]
        probabilities = {LABEL_MAP[j]: float(pred[j]) for j in range(len(LABEL_MAP))}

        results.append({
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities
        })

    return results
