"""
Improved Brain Tumor CNN Training
Simple, student-friendly approach with essential anti-overfitting techniques
Target: 90-95% test accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# ============================================================================
# 1. Configuration
# ============================================================================
TRAIN_DIR = Path("../../data/brain-tumor-mri-preproc/Training")
TEST_DIR = Path("../../data/brain-tumor-mri-preproc/Testing")
IMG_SIZE = (192, 192)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

# ============================================================================
# 2. Data Augmentation (KEY for preventing overfitting)
# ============================================================================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
], name="augmentation")

# ============================================================================
# 3. Load Datasets
# ============================================================================
# Training set WITH augmentation
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
)

# Validation set WITHOUT augmentation
val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
)

# Test set
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

# ============================================================================
# 4. Build Model (Simple but effective)
# ============================================================================
inputs = keras.Input(shape=IMG_SIZE + (1,))

# Apply augmentation only during training
x = data_augmentation(inputs)

# Normalize pixel values to [0, 1]
x = layers.Rescaling(1./255)(x)

# CNN layers
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

# Classifier
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)  # KEY: Prevent overfitting
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)  # KEY: Prevent overfitting
outputs = layers.Dense(len(train_ds.class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs, name="brain_tumor_cnn")
model.summary()

# ============================================================================
# 5. Compile
# ============================================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ============================================================================
# 6. Callbacks (Essential for good performance)
# ============================================================================
callbacks = [
    # Stop when val_loss stops improving (KEY: prevents overtraining)
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce learning rate when stuck
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
]

# ============================================================================
# 7. Train
# ============================================================================
print("\n" + "="*70)
print("TRAINING")
print("="*70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ============================================================================
# 8. Evaluate on Test Set
# ============================================================================
print("\n" + "="*70)
print("FINAL EVALUATION ON TEST SET")
print("="*70)

test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"✅ Final Test Loss: {test_loss:.4f}")

# ============================================================================
# 9. Save Model and Training History
# ============================================================================
export_dir = "brain_tumor_cnn_improved"
model.export(export_dir)
print(f"\n✅ Model saved to: {export_dir}")

# Save training history
import json
history_path = f"{export_dir}/training_history.json"
history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"✅ Training history saved to: {history_path}")

# ============================================================================
# 10. Training Summary
# ============================================================================
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"Final Training Accuracy:   {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"Final Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"\nGap (Train - Val): {(final_train_acc - final_val_acc)*100:.2f}%")
print(f"Gap (Val - Test):  {(final_val_acc - test_acc)*100:.2f}%")

if final_train_acc - final_val_acc > 0.05:
    print("\n⚠️  Still some overfitting (>5% gap). Consider:")
    print("   - More data augmentation")
    print("   - Higher dropout rates")
else:
    print("\n✅ Good generalization - minimal overfitting!")

if test_acc >= 0.90:
    print(f"✅ TARGET ACHIEVED: {test_acc*100:.2f}% >= 90%")
else:
    print(f"⚠️  Below target: {test_acc*100:.2f}% < 90%")
