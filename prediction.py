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
KERAS_MODEL_PATH = Path(__file__).parent / "notebooks" / "exploration" / "brain_tumor_cnn_keras"

LABEL_MAP = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary"
}

TARGET_SIZE = (192, 192)
CLIP_PCT = (1, 99)

_model = None
_keras_model = None

def load_model():
    """Load and cache the trained model"""
    global _model
    if _model is None:
        loaded = tf.saved_model.load(str(MODEL_PATH))
        _model = loaded.signatures['serving_default']
    return _model

def load_keras_model():
    """Load and cache the model as a Keras model for Grad-CAM"""
    global _keras_model
    if _keras_model is None:
        # Load the Keras model directly
        _keras_model = keras.models.load_model(str(KERAS_MODEL_PATH))
    return _keras_model

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

def get_last_conv_layer_name(model):
    """Find the last convolutional layer in the model"""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    return None

def generate_gradcam(image: np.ndarray, class_idx: int = None) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for the given image

    Args:
        image: Original image (preprocessed will be done internally)
        class_idx: Target class index for Grad-CAM (if None, use predicted class)

    Returns:
        Heatmap as numpy array (same size as preprocessed image)
    """
    model = load_keras_model()
    processed = preprocess_image(image)
    img_array = np.expand_dims(processed, axis=0)

    # Find the last convolutional layer
    last_conv_layer_name = get_last_conv_layer_name(model)
    if last_conv_layer_name is None:
        raise ValueError("No convolutional layer found in model")

    # Create a model that maps the input to the activations of the last conv layer and the output predictions
    grad_model = keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        class_channel = predictions[:, class_idx]

    # Get the gradients of the output with respect to the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)

    # Compute the guided gradients (global average pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps by the computed gradients
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Create the heatmap
    heatmap = np.mean(conv_outputs, axis=-1)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap

def apply_gradcam_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image

    Args:
        image: Original image (RGB or grayscale)
        heatmap: Grad-CAM heatmap (from generate_gradcam)
        alpha: Transparency of the heatmap overlay (0-1)

    Returns:
        Image with heatmap overlay as RGB numpy array
    """
    # Convert original image to RGB if needed
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = image.copy()

    # Resize heatmap to match original image size
    if img_rgb.shape[:2] != heatmap.shape:
        heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    else:
        heatmap_resized = heatmap

    # Convert heatmap to RGB colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Ensure image is in correct format
    if img_rgb.dtype != np.uint8:
        img_rgb = (img_rgb * 255).astype(np.uint8) if img_rgb.max() <= 1.0 else img_rgb.astype(np.uint8)

    # Overlay the heatmap
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay
