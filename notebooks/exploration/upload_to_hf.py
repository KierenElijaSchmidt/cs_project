#!/usr/bin/env python3
"""Upload NeuroSight models to Hugging Face"""

from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(dotenv_path="../../.env")

# Configuration
token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    raise ValueError("HUGGINGFACE_TOKEN not found in .env file")

username = "kierenschmidthsg"
repo_name = "neurosight-brain-tumor-models"
repo_id = f"{username}/{repo_name}"

# Initialize API
api = HfApi(token=token)

print(f"Creating repository: {repo_id}")
try:
    create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="model",
        exist_ok=True,
        private=False
    )
    print("✓ Repository created/verified")
except Exception as e:
    print(f"Repository creation: {e}")

# Upload the compressed models
print("\nUploading model archive...")
archive_path = "neurosight-models.tar.gz"

try:
    api.upload_file(
        path_or_fileobj=archive_path,
        path_in_repo="neurosight-models.tar.gz",
        repo_id=repo_id,
        token=token
    )
    print("✓ Models uploaded successfully!")
    print(f"\nYour models are now available at:")
    print(f"https://huggingface.co/{repo_id}")
    print(f"\nDirect download link:")
    print(f"https://huggingface.co/{repo_id}/resolve/main/neurosight-models.tar.gz")
except Exception as e:
    print(f"✗ Upload failed: {e}")

# Create a README for the model repository
readme_content = """---
license: cc0-1.0
tags:
- medical-imaging
- brain-tumor
- cnn
- tensorflow
- computer-vision
---

# NeuroSight Brain Tumor Classification Models

Pre-trained TensorFlow/Keras models for classifying brain MRI scans into four categories:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor

## Model Details

- **Architecture**: Custom CNN with 4 convolutional layers
- **Framework**: TensorFlow/Keras
- **Input**: 192×192 grayscale MRI images
- **Test Accuracy**: ~91%
- **Test Recall**: ~90%

## Usage

Download and extract the models:

```bash
wget https://huggingface.co/{}/resolve/main/neurosight-models.tar.gz
tar -xzf neurosight-models.tar.gz
```

This creates two directories:
- `brain_tumor_cnn_improved/` - Main classification model
- `brain_tumor_cnn_keras/` - Model for Grad-CAM visualizations

## Training Dataset

Trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.

## Repository

Full project code: [GitHub Repository](<your-github-url>)

## License

CC0 1.0 Universal (same as training dataset)

## Disclaimer

This is an educational project and should not be used for medical diagnosis.
""".format(repo_id)

print("\nCreating README...")
try:
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token
    )
    print("✓ README created")
except Exception as e:
    print(f"README creation: {e}")

print("\n✓ All done!")
