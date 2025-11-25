"""
Quick evaluation script for the improved model
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load test dataset
test_ds = keras.utils.image_dataset_from_directory(
    '../../data/brain-tumor-mri-preproc/Testing',
    image_size=(192, 192),
    color_mode='grayscale',
    batch_size=32,
    shuffle=False
)

# Load improved model using TFSMLayer
model_path = 'brain_tumor_cnn_improved'
reloaded = tf.saved_model.load(model_path)
infer = reloaded.signatures['serving_default']

# Get predictions
all_labels = []
all_preds = []

for images, labels in test_ds:
    # Predictions
    preds = infer(input_1=images)['output_0']
    pred_classes = tf.argmax(preds, axis=1)

    all_labels.extend(labels.numpy())
    all_preds.extend(pred_classes.numpy())

# Calculate accuracy
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
accuracy = np.mean(all_labels == all_preds)

print(f'\n========== FINAL RESULTS ==========')
print(f'Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
print(f'===================================')

if accuracy >= 0.90:
    print(f'✅ TARGET ACHIEVED: {accuracy*100:.2f}% >= 90%')
else:
    print(f'⚠️  Below target: {accuracy*100:.2f}% < 90%')
