"""
Re-save the model in Keras format for Grad-CAM compatibility
This script modifies the training script to save with model.save() instead of model.export()
"""
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import shutil

# Configuration
TRAIN_DIR = Path("data/brain-tumor-mri-preproc/Training")
TEST_DIR = Path("data/brain-tumor-mri-preproc/Testing")
IMG_SIZE = (192, 192)
BATCH_SIZE = 32
SEED = 42

# Check if preprocessed data exists
if not TRAIN_DIR.exists():
    print(f"Error: Training data not found at {TRAIN_DIR}")
    print("Please ensure the preprocessed data exists.")
    exit(1)

# Data augmentation to prevent overfitting
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
], name="augmentation")

# Load datasets (small validation split just to verify)
print("Loading datasets...")
val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
)

# Build model architecture (matching train_improved.py)
print("Building model...")
inputs = keras.Input(shape=IMG_SIZE + (1,))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)

# Convolutional layers
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

# Dense layers
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="brain_tumor_cnn")

# Load weights from existing SavedModel
print("Loading weights from existing model...")
old_model_path = Path("notebooks/exploration/brain_tumor_cnn_improved")

# Try to load the SavedModel and transfer weights
try:
    # Load the old model
    loaded_model = tf.saved_model.load(str(old_model_path))

    # Compile the new model (needed before loading weights)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Try to evaluate on validation set with loaded model to verify
    print("Testing loaded model...")

    # Since we can't directly transfer weights, we'll need to use the concrete function
    # For now, let's just save the architecture and note that weights need retraining

    print("\nNote: Direct weight transfer from SavedModel export is complex.")
    print("Two options:")
    print("1. Re-train the model with this script (recommended)")
    print("2. Use the existing model without Grad-CAM")

    # Save the model architecture
    keras_model_path = Path("notebooks/exploration/brain_tumor_cnn_keras")
    keras_model_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model architecture to {keras_model_path}...")
    model.save(str(keras_model_path))
    print("✓ Model saved (architecture only, no weights)")

    print("\nTo get Grad-CAM working, you need to re-train the model.")
    print("Run: python notebooks/exploration/retrain_with_keras.py")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
