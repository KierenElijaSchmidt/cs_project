"""
Test Grad-CAM functionality
"""
import numpy as np
from PIL import Image
from pathlib import Path
from prediction import generate_gradcam, apply_gradcam_overlay, predict_single

# Get a test image
TEST_DATA_PATH = Path("data/brain-tumor-mri-dataset/Testing")

# Find first image from each class
for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
    class_dir = TEST_DATA_PATH / class_name
    if class_dir.exists():
        img_path = list(class_dir.glob("*.jpg"))[0]
        print(f"\nTesting with {class_name}: {img_path.name}")

        # Load image
        img = Image.open(img_path)
        img_array = np.array(img)

        # Get prediction
        result = predict_single(img_array)
        print(f"  Predicted: {result['label']} ({result['confidence']:.1%})")

        # Generate Grad-CAM
        try:
            print("  Generating Grad-CAM...")
            heatmap = generate_gradcam(img_array)
            print(f"  ✓ Heatmap generated (shape: {heatmap.shape})")

            # Apply overlay
            gradcam_img = apply_gradcam_overlay(img_array, heatmap, alpha=0.5)
            print(f"  ✓ Overlay applied (shape: {gradcam_img.shape})")

            # Save visualization
            output_path = f"gradcam_test_{class_name}.jpg"
            Image.fromarray(gradcam_img).save(output_path)
            print(f"  ✓ Saved to: {output_path}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

        break  # Just test one image for now

print("\n✓ Grad-CAM test complete!")
