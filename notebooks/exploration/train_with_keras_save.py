"""
Brain Tumor CNN Training Script with Weighted Recall
Trains a CNN to classify MRI scans into 4 categories
Uses weighted recall to prioritize critical classes (glioma, notumor)
Saves in Keras format for Grad-CAM compatibility
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# Medical Priority Weights
# Based on clinical severity and importance of correct detection
CLASS_WEIGHTS = {
    0: 3.0,   # glioma (malignant, aggressive - CRITICAL)
    1: 1.5,   # meningioma (usually benign)
    2: 3.0,   # notumor (must not miss cancer - CRITICAL)
    3: 2.0    # pituitary (serious but manageable)
}

# Corresponding weights for weighted recall calculation
RECALL_WEIGHTS = [3.0, 1.5, 3.0, 2.0]  # [glioma, meningioma, notumor, pituitary]


# Custom Weighted Recall Metric
class WeightedRecall(keras.metrics.Metric):
    """
    Weighted recall that prioritizes critical classes.

    Formula: Sum(weight_i * recall_i) / Sum(weight_i)

    Where:
    - recall_i = true_positives_i / (true_positives_i + false_negatives_i)
    - weight_i = clinical importance weight for class i
    """
    def __init__(self, num_classes=4, class_weights=None, name='weighted_recall', **kwargs):
        super(WeightedRecall, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.class_weights = class_weights if class_weights is not None else [1.0] * num_classes

        # Create per-class true positive and false negative counters
        self.true_positives = [self.add_weight(
            name=f'tp_{i}', initializer='zeros', dtype=tf.float32
        ) for i in range(num_classes)]

        self.false_negatives = [self.add_weight(
            name=f'fn_{i}', initializer='zeros', dtype=tf.float32
        ) for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int64)
        y_pred_classes = tf.cast(y_pred_classes, tf.int64)

        for i in range(self.num_classes):
            # True positives: predicted class i AND actual class i
            tp = tf.reduce_sum(
                tf.cast(tf.logical_and(
                    tf.equal(y_true, i),
                    tf.equal(y_pred_classes, i)
                ), tf.float32)
            )

            # False negatives: actual class i BUT predicted something else
            fn = tf.reduce_sum(
                tf.cast(tf.logical_and(
                    tf.equal(y_true, i),
                    tf.not_equal(y_pred_classes, i)
                ), tf.float32)
            )

            self.true_positives[i].assign_add(tp)
            self.false_negatives[i].assign_add(fn)

    def result(self):
        weighted_recall_sum = 0.0
        weight_sum = 0.0

        for i in range(self.num_classes):
            recall = self.true_positives[i] / (self.true_positives[i] + self.false_negatives[i] + 1e-7)
            weighted_recall_sum += self.class_weights[i] * recall
            weight_sum += self.class_weights[i]

        return weighted_recall_sum / weight_sum

    def reset_state(self):
        for i in range(self.num_classes):
            self.true_positives[i].assign(0)
            self.false_negatives[i].assign(0)

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
# Using Weighted Recall as primary metric (prioritizes glioma and notumor)
# Accuracy is included for reference
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=[
        WeightedRecall(class_weights=RECALL_WEIGHTS, name='weighted_recall'),  # Primary metric
        "accuracy",  # Secondary metric for reference
    ],
)

print("\n" + "="*60)
print("MEDICAL PRIORITY WEIGHTS:")
print("  Glioma (malignant):     3.0x weight")
print("  Notumor (don't miss):   3.0x weight")
print("  Pituitary (serious):    2.0x weight")
print("  Meningioma (benign):    1.5x weight")
print("="*60 + "\n")

# Callbacks for training optimization
# Monitor validation weighted recall (higher is better) for early stopping
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_weighted_recall",
        mode="max",  # Maximize weighted recall (higher is better)
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",  # Keep monitoring loss for learning rate reduction
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
]

# Train
print("\nTraining model with class weights...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=CLASS_WEIGHTS  # Apply class weights to loss function
)

# Evaluate
print("\nEvaluating on test set...")
test_results = model.evaluate(test_ds)
test_loss = test_results[0]
test_weighted_recall = test_results[1]  # Weighted recall is first metric
test_acc = test_results[2]              # Accuracy is second metric
print(f"Test weighted recall: {test_weighted_recall:.4f} ({test_weighted_recall*100:.2f}%)")
print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test loss: {test_loss:.4f}")

# Save model in Keras format (NOT export, which creates SavedModel)
export_dir = "brain_tumor_cnn_keras"
print(f"\nSaving model in Keras format to: {export_dir}")
model.save(export_dir)  # Use save() instead of export() for Grad-CAM compatibility
print(f"✓ Model saved to: {export_dir}")

# Also save in SavedModel format for backward compatibility
export_dir_sm = "brain_tumor_cnn_savedmodel"
model.export(export_dir_sm)
print(f"✓ SavedModel saved to: {export_dir_sm}")

# Save training history for visualization
history_path = f"{export_dir}/training_history.json"
history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"✓ Training history saved to: {history_path}")

# Print summary
final_train_weighted_recall = history.history['weighted_recall'][-1]
final_val_weighted_recall = history.history['val_weighted_recall'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"\n{'='*60}")
print(f"=== Primary Metric: Weighted Recall ===")
print(f"{'='*60}")
print(f"Training weighted recall: {final_train_weighted_recall:.4f} ({final_train_weighted_recall*100:.2f}%)")
print(f"Validation weighted recall: {final_val_weighted_recall:.4f} ({final_val_weighted_recall*100:.2f}%)")
print(f"Test weighted recall: {test_weighted_recall:.4f} ({test_weighted_recall*100:.2f}%)")
print(f"Train-val gap: {(final_train_weighted_recall - final_val_weighted_recall)*100:.2f}%")

print(f"\n{'='*60}")
print(f"=== Secondary Metric: Accuracy ===")
print(f"{'='*60}")
print(f"Training accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"Validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

if final_train_weighted_recall - final_val_weighted_recall > 0.05:
    print("\nWarning: Weighted recall gap > 5%, possible overfitting")
else:
    print("\nGood generalization")

if test_weighted_recall >= 0.90:
    print(f"Target achieved: {test_weighted_recall*100:.2f}% weighted recall >= 90%")
else:
    print(f"Below target: {test_weighted_recall*100:.2f}% weighted recall < 90%")

print("\n" + "="*60)
print("IMPORTANT: To use this model with Grad-CAM:")
print(f"  Update MODEL_PATH in prediction.py to point to: {export_dir}")
print("="*60)
