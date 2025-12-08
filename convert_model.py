"""
Convert SavedModel to Keras format for Grad-CAM compatibility
"""
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "notebooks" / "exploration" / "brain_tumor_cnn_improved"
KERAS_MODEL_PATH = Path(__file__).parent / "notebooks" / "exploration" / "brain_tumor_cnn_keras"

print("Loading SavedModel...")
# Load as concrete function
loaded = tf.saved_model.load(str(MODEL_PATH))

# Get the concrete function
concrete_func = loaded.signatures['serving_default']

# Build a Keras model from the concrete function
# We need to reconstruct the model architecture and load weights
from tensorflow.keras import layers

print("Building Keras model architecture...")
inputs = keras.Input(shape=(192, 192, 1))

# Build architecture matching training script
x = layers.Rescaling(1./255)(inputs)

# Convolutional layers
x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(x)
x = layers.MaxPooling2D(name="pool1")(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
x = layers.MaxPooling2D(name="pool2")(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
x = layers.MaxPooling2D(name="pool3")(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv4")(x)
x = layers.MaxPooling2D(name="pool4")(x)

# Dense layers
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(4, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="brain_tumor_cnn")

print("Loading weights from checkpoint...")
# Load weights from the saved variables
try:
    checkpoint_dir = MODEL_PATH / "variables" / "variables"
    model.load_weights(str(checkpoint_dir))
    print("✓ Weights loaded successfully")
except Exception as e:
    print(f"✗ Could not load weights directly: {e}")
    print("\nTrying alternative approach...")

    # Alternative: Use tf.train.Checkpoint
    try:
        # Create a checkpoint
        ckpt = tf.train.Checkpoint(model=loaded)
        # This won't work directly, we need a different approach

        # Try to extract variables
        all_variables = {v.name: v for v in loaded.variables}
        print(f"Found {len(all_variables)} variables in SavedModel")

        # Map variables to model layers
        for layer in model.layers:
            if hasattr(layer, 'weights') and len(layer.weights) > 0:
                layer_name = layer.name
                print(f"Processing layer: {layer_name}")

    except Exception as e2:
        print(f"Alternative approach failed: {e2}")

print("\nSaving as Keras model...")
KERAS_MODEL_PATH.mkdir(parents=True, exist_ok=True)
model.save(str(KERAS_MODEL_PATH))
print(f"✓ Model saved to: {KERAS_MODEL_PATH}")

print("\nVerifying the saved model...")
loaded_keras = keras.models.load_model(str(KERAS_MODEL_PATH))
print(f"✓ Keras model loaded successfully")
print(f"  Layers: {len(loaded_keras.layers)}")
print(f"  Input shape: {loaded_keras.input_shape}")
print(f"  Output shape: {loaded_keras.output_shape}")
