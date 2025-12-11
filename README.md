# NeuroSight - Brain Tumor MRI Classification

A deep learning application for classifying brain MRI scans into four categories: glioma, meningioma, pituitary tumor, and no tumor. Built with TensorFlow and Streamlit, achieving ~90% test accuracy.

**The trained model is available on [Hugging Face](https://huggingface.co/kierenschmidthsg/neurosight-brain-tumor-models) so you don't need to train it yourself.**

## Features

- CNN-based classification with custom architecture
- Interactive Streamlit web interface for uploading and analyzing MRI scans
- Model performance analysis with confusion matrices and recall metrics
- Grad-CAM visualizations showing which image regions influence predictions
- AI-assisted medical report generation using Claude API
- Training progress visualization with learning curves
- Built-in data labeling interface for expanding the training set

## Tech Stack

- TensorFlow/Keras for deep learning
- Streamlit for the web interface
- OpenCV and PIL for image processing
- Anthropic Claude API for report generation
- Plotly and Matplotlib for visualizations
- NumPy and Pandas for data processing

## Installation

**Note:** The trained models are not included in this repository due to file size. Download them from [Hugging Face](https://huggingface.co/kierenschmidthsg/neurosight-brain-tumor-models) where they have been uploaded for you, or train them yourself (see Model Setup section).

### Prerequisites

- Python 3.8 or higher (tested with 3.8.12)
- ~3-4 GB free disk space (for dataset and models)
- Kaggle account (to download the dataset)

### Environment Setup

Clone the repository:
```bash
git clone <repository-url>
cd cs_project
```

Using make (Linux/macOS):
```bash
make all
```

Manual setup (Windows or if make is unavailable):
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### API Key Configuration

To use the AI report generation feature, create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

Get your API key from [Anthropic Console](https://console.anthropic.com/). Note that this is optional - the classification features work without it.

### Dataset Setup

The dataset is not included in this repository due to its size (~2.8 GB). Download it from Kaggle:

Using Kaggle CLI:
```bash
pip install kaggle
# Set up API credentials at ~/.kaggle/kaggle.json (see https://www.kaggle.com/docs/api)
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/brain-tumor-mri-dataset
rm brain-tumor-mri-dataset.zip
```

Manual download:
1. Visit [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
2. Download the dataset
3. Extract to `data/brain-tumor-mri-dataset/`

Expected directory structure:
```
data/brain-tumor-mri-dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### Model Setup

The trained models are required to run the application but are not included in the repository. You have two options:

**Option 1: Download from Hugging Face (Recommended)**

The trained models have been uploaded to Hugging Face for easy access. Download them with:

```bash
cd notebooks/exploration
wget https://huggingface.co/kierenschmidthsg/neurosight-brain-tumor-models/resolve/main/neurosight-models.tar.gz
tar -xzf neurosight-models.tar.gz
rm neurosight-models.tar.gz
```

This extracts the models to `brain_tumor_cnn_improved/` and `brain_tumor_cnn_keras/`.

**Option 2: Train from Scratch**

If you want to train the models yourself (~20-30 minutes on CPU):

```bash
# Preprocess the data first
cd notebooks/exploration
jupyter notebook Preproc.ipynb  # Run all cells

# Train the model
python train_improved.py
```

**Verify Setup:**
```bash
ls data/brain-tumor-mri-dataset/Testing/
ls notebooks/exploration/brain_tumor_cnn_improved/
ls notebooks/exploration/brain_tumor_cnn_keras/
```

## Usage

Start the application:
```bash
streamlit run app.py
```

The web interface opens at `http://localhost:8501` with three tabs:

**Upload Images** - Upload MRI scans for classification, view confidence scores, generate Grad-CAM visualizations, and create AI reports

**Test Model Performance** - Evaluate the model on test data, view confusion matrices, analyze recall metrics, and explore learning curves

**About** - Project information and tumor type descriptions

### Retraining

To retrain from scratch:
```bash
cd notebooks/exploration
python train_improved.py
```

To evaluate:
```bash
cd notebooks/exploration
python eval_improved.py
```

## Project Structure

```
cs_project/
├── app.py                          # Main Streamlit application
├── prediction.py                   # Model loading and inference
├── requirements.txt                # Dependencies
├── Makefile                        # Setup automation
├── notebooks/
│   └── exploration/
│       ├── Preproc.ipynb          # Data preprocessing
│       ├── train_improved.py      # Training script
│       └── eval_improved.py       # Evaluation script
└── data/
    ├── brain-tumor-mri-dataset/   # Original Kaggle dataset
    ├── brain-tumor-mri-preproc/   # Preprocessed images
    └── labeled_training/          # User-labeled images
```

## Model Architecture

The CNN architecture:
- 4 convolutional layers with 32, 64, 128, and 128 filters
- Max pooling after each convolutional layer
- 2 fully connected layers with 256 and 128 neurons
- Dropout layers (0.5 and 0.3) for regularization
- Softmax output layer for 4 classes

Training configuration:
- Input: 192×192 grayscale images
- Batch size: 32
- Optimizer: Adam (learning rate 0.001)
- Data augmentation: horizontal flips, ±15° rotation, ±10% zoom
- Early stopping with patience of 5 epochs
- Learning rate reduction on plateau

Performance metrics:
- Test accuracy: ~91%
- Test recall (macro): ~90%
- Training epochs: ~20
- Model size: ~36 MB

Recall was prioritized as the primary metric to ensure tumor cases are not missed.

## Dataset

The project uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.

Class distribution:
- Glioma: 1,321 training / 300 test images
- Meningioma: 1,339 training / 306 test images
- No Tumor: 1,595 training / 405 test images
- Pituitary: 1,457 training / 300 test images

Preprocessing steps:
1. Grayscale conversion
2. Resize to 192×192 pixels
3. Percentile clipping (1st to 99th percentile)
4. Min-max normalization to [0, 1]
5. Save as 8-bit PNG

## Troubleshooting

**Import errors (tensorflow, etc.)**
- Activate your virtual environment
- Reinstall: `pip install -r requirements.txt`

**FileNotFoundError for dataset**
- Download the dataset following the Dataset Setup section
- Check that the folder structure matches the expected layout

**SavedModel file does not exist**
- Train the models using `python train_improved.py` in `notebooks/exploration/`

**API key not configured warning**
- Only needed for AI report generation
- Other features work without the API key

**Training is slow or runs out of memory**
- Reduce batch size in `train_improved.py` (change `batch_size=32` to `batch_size=16`)
- Training takes 20-30 minutes with GPU, 2-4 hours with CPU

**make command not found (Windows)**
- Use the manual setup instructions instead
- Or use WSL (Windows Subsystem for Linux)

## Makefile Commands

```bash
make env        # Set up Python environment with pyenv
make install    # Install dependencies
make clean      # Remove cache and build files
make all        # Complete setup (env + install)
```

## Limitations

This is an educational project and should not be used for medical diagnosis. The model is trained on a limited dataset and may not generalize to all cases. Any real-world application would require validation by qualified medical professionals.

## Future Work

- Expand training dataset using the labeling interface
- Add precision and F1-score metrics
- Implement batch processing API
- Deploy as containerized web service
- Support additional MRI sequences (T1, T2, FLAIR)
- Add DICOM format support

## Acknowledgments

- Dataset: Masoud Nickparvar on Kaggle
- Trained models hosted on [Hugging Face](https://huggingface.co/kierenschmidthsg/neurosight-brain-tumor-models)
- AI integration: Anthropic Claude API
- Frameworks: Streamlit and TensorFlow

## License

Educational purposes only. Dataset license: CC0 1.0 Universal.
