import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# Model path - use SavedModel format for TensorFlow 2.x compatibility
MODEL_PATH = Path(__file__).parent / "notebooks" / "exploration" / "brain_tumor_cnn_improved"

# Label mapping
LABEL_MAP = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary"
}

# Preprocessing settings (from notebook)
TARGET_SIZE = (192, 192)  # Model was trained with 192x192
CLIP_PCT = (1, 99)

# Global model cache
_model = None

def load_model():
    """Load the Keras model (cached)."""
    global _model
    if _model is None:
        loaded = tf.saved_model.load(str(MODEL_PATH))
        # Get the serving signature
        _model = loaded.signatures['serving_default']
    return _model

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess a single image for prediction.

    Args:
        image: Input image as numpy array (can be grayscale or RGB)

    Returns:
        Preprocessed image ready for model input
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        img = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        img = image

    # Convert to float32
    img = img.astype(np.float32)

    # Resize
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # Percentile clipping and normalization
    lo, hi = np.percentile(img, CLIP_PCT)
    img = np.clip(img, lo, hi)
    denom = (hi - lo) if hi > lo else 1.0
    img = (img - lo) / denom

    # Convert grayscale to RGB (3 channels) for EfficientNet
    img = np.stack([img, img, img], axis=-1)

    # Apply EfficientNet preprocessing
    img = img * 255.0  # Back to 0-255 range
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return img

def predict_single(image: np.ndarray) -> dict:
    """
    Predict tumor type for a single image.

    Args:
        image: Input image as numpy array

    Returns:
        Dictionary with prediction results
    """
    model = load_model()

    # Preprocess
    processed = preprocess_image(image)

    # Add batch dimension
    batch = np.expand_dims(processed, axis=0)

    # Predict - SavedModel signature returns a dict
    output = model(tf.constant(batch, dtype=tf.float32))
    # Get the first (and likely only) output tensor
    output_key = list(output.keys())[0]
    predictions = output[output_key].numpy()

    # Get results
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx])
    label = LABEL_MAP[class_idx]

    # All class probabilities
    probabilities = {LABEL_MAP[i]: float(predictions[0][i]) for i in range(len(LABEL_MAP))}

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities
    }

def predict_batch(images: list) -> list:
    """
    Predict tumor types for multiple images.

    Args:
        images: List of input images as numpy arrays

    Returns:
        List of prediction result dictionaries
    """
    model = load_model()

    # Preprocess all images
    processed = np.array([preprocess_image(img) for img in images])

    # Predict - SavedModel signature returns a dict
    output = model(tf.constant(processed, dtype=tf.float32))
    # Get the first (and likely only) output tensor
    output_key = list(output.keys())[0]
    predictions = output[output_key].numpy()

    # Get results for each image
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
