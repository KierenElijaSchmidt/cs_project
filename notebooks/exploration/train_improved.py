"""
Brain Tumor CNN Training Script
Trains a CNN to classify MRI scans into 4 categories
"""

import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# Configuration
TRAIN_DIR = Path("../../data/brain-tumor-mri-preproc/Training")
TEST_DIR = Path("../../data/brain-tumor-mri-preproc/Testing")
IMG_SIZE = (192, 192)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

# Data augmentation to prevent overfitting
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
], name="augmentation")

# Load datasets
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
)

val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Classes:", train_ds.class_names)
print(f"Training batches: {len(train_ds)}")
print(f"Validation batches: {len(val_ds)}")
print(f"Test batches: {len(test_ds)}")

# Build model
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
outputs = layers.Dense(len(train_ds.class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs, name="brain_tumor_cnn")
model.summary()

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Callbacks for training optimization
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
]

# Train
print("\nTraining model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Evaluate
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test loss: {test_loss:.4f}")

# Save model
export_dir = "brain_tumor_cnn_improved"
model.export(export_dir)
print(f"Model saved to: {export_dir}")

# Save training history for visualization
history_path = f"{export_dir}/training_history.json"
history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"Training history saved to: {history_path}")

# Print summary
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"\nTraining accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"Validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Train-val gap: {(final_train_acc - final_val_acc)*100:.2f}%")

if final_train_acc - final_val_acc > 0.05:
    print("Warning: Gap > 5%, possible overfitting")
else:
    print("Good generalization")

if test_acc >= 0.90:
    print(f"Target achieved: {test_acc*100:.2f}% >= 90%")
else:
    print(f"Below target: {test_acc*100:.2f}% < 90%")
