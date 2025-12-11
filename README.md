# NeuroSight - Brain Tumor MRI Classification

A deep learning application for classifying brain MRI scans into four categories: glioma, meningioma, pituitary tumor, and no tumor. Built with TensorFlow and Streamlit.

## Features

- **CNN-based Classification**: Custom convolutional neural network achieving 90%+ test accuracy
- **Web Interface**: Interactive Streamlit dashboard for uploading and classifying MRI scans
- **Model Performance Analysis**: Test the model on random samples with confusion matrix, recall metrics, and per-class performance
- **Grad-CAM Visualization**: Visual explanations showing which regions the model focuses on
- **AI-Assisted Reports**: Automatic medical report generation using Claude AI
- **Learning Curves**: Visualization of training progress and model performance
- **Data Labeling**: Built-in interface for labeling new images to expand training data

## Tech Stack

- **Deep Learning**: TensorFlow/Keras
- **Web Framework**: Streamlit
- **Computer Vision**: OpenCV, PIL
- **AI Integration**: Anthropic Claude API
- **Visualization**: Plotly, Matplotlib
- **Data Processing**: NumPy, Pandas

## Installation

### Prerequisites

- **Python 3.8.12** (recommended, or Python 3.8+)
- **pip** package manager
- **Git** version control
- **~3-4 GB free disk space** (dataset + models + dependencies)
- **Kaggle account** (for dataset download)
- **Operating System**: Linux, macOS, or Windows with WSL (Windows native support via manual setup only)

### Quick Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd cs_project
```

2. Set up environment and install dependencies:
```bash
make all
```

This will:
- Install Python 3.8.12 via pyenv
- Create a virtual environment named `cspyenv`
- Install all dependencies from `requirements.txt`

### Manual Setup

If you prefer manual setup or don't have `make` installed:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Windows Users (without WSL):**
- The `make` command won't work natively on Windows
- Use the manual setup above
- Ensure you have Python 3.8+ installed from [python.org](https://www.python.org/downloads/)
- Use Command Prompt or PowerShell (replace `source` with the Windows activate command)

### Environment Configuration

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

Get your API key from [Anthropic Console](https://console.anthropic.com/).

### Dataset Setup

**Important**: The dataset is **not included** in this repository due to its size (~2.8 GB). You must download it separately.

#### Option 1: Kaggle CLI (Recommended)

1. Install Kaggle CLI:
```bash
pip install kaggle
```

2. Set up Kaggle API credentials:
   - Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Scroll to "API" section and click "Create New Token"
   - This downloads `kaggle.json` - place it in:
     - Linux/Mac: `~/.kaggle/kaggle.json`
     - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - Set permissions (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

3. Download and extract the dataset:
```bash
# From project root directory
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/brain-tumor-mri-dataset
rm brain-tumor-mri-dataset.zip  # Clean up zip file
```

#### Option 2: Manual Download

1. Visit [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
2. Click "Download" button (requires Kaggle account)
3. Extract the ZIP file into `data/brain-tumor-mri-dataset/` in your project directory
4. Verify the structure:
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

### Model Files Setup

**Important**: The trained models are required to run the application.

#### Option 1: Train from Scratch (Recommended for Learning)

If you want to train the model yourself (~20-30 minutes with CPU):

```bash
# First, preprocess the data
cd notebooks/exploration
jupyter notebook Preproc.ipynb  # Run all cells

# Then train the model
python train_improved.py

# This creates:
# - notebooks/exploration/brain_tumor_cnn_improved/ (main model)
# - notebooks/exploration/brain_tumor_cnn_keras/ (for Grad-CAM)
```

#### Option 2: Use Pre-trained Models

If pre-trained models are provided separately (via Google Drive, Hugging Face, etc.):
1. Download the model files
2. Extract them to:
   - `notebooks/exploration/brain_tumor_cnn_improved/`
   - `notebooks/exploration/brain_tumor_cnn_keras/`

**Note**: If models are not provided and you skip training, the application will fail to start.

### Complete Setup Checklist

Before running the application, ensure you have:

- [x] ✅ Cloned the repository
- [x] ✅ Installed Python 3.8+ and dependencies (`pip install -r requirements.txt`)
- [x] ✅ Created `.env` file with `ANTHROPIC_API_KEY` (optional, needed for AI reports)
- [x] ✅ Downloaded dataset from Kaggle to `data/brain-tumor-mri-dataset/`
- [x] ✅ Trained models exist in `notebooks/exploration/brain_tumor_cnn_improved/` and `brain_tumor_cnn_keras/`

**Quick verification:**
```bash
# Check if dataset exists
ls data/brain-tumor-mri-dataset/Testing/

# Check if models exist
ls notebooks/exploration/brain_tumor_cnn_improved/
ls notebooks/exploration/brain_tumor_cnn_keras/

# If models missing, train them:
cd notebooks/exploration && python train_improved.py
```

## Usage

### Running the Application

Start the Streamlit web interface:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Application Tabs

**1. Upload Images**
- Upload 1-10 MRI scans (PNG, JPG, JPEG)
- Get instant classification results with confidence scores
- Optional Grad-CAM visualization to see model focus areas
- Generate AI-assisted medical reports
- Save labeled images to improve the model

**2. Test Model Performance**
- Select random images from the test dataset
- View predictions vs actual labels
- Analyze confusion matrix showing prediction patterns
- Review recall metrics (primary metric) and per-class performance
- Explore training learning curves
- Chat with AI assistant about model behavior

**3. About**
- Project overview and motivation
- Tumor type descriptions
- How the system works

### Training the Model

To retrain the model from scratch:

```bash
cd notebooks/exploration
python train_improved.py
```

This will:
- Load preprocessed training data
- Train a CNN with data augmentation
- Apply early stopping and learning rate reduction
- Save the model to `brain_tumor_cnn_improved/`
- Export training history as JSON

### Evaluating the Model

Quick evaluation on test set:

```bash
cd notebooks/exploration
python eval_improved.py
```

## Project Structure

```
cs_project/
├── app.py                          # Main Streamlit application
├── prediction.py                   # Model loading and inference
├── requirements.txt                # Python dependencies
├── Makefile                        # Setup automation
├── .env                           # Environment variables (not in git)
│
├── notebooks/
│   └── exploration/
│       ├── Preproc.ipynb          # Data preprocessing notebook
│       ├── train_improved.py      # Model training script
│       ├── eval_improved.py       # Model evaluation script
│       └── brain_tumor_cnn_improved/  # Trained model
│
└── data/
    ├── brain-tumor-mri-dataset/   # Original dataset
    ├── brain-tumor-mri-preproc/   # Preprocessed images
    └── labeled_training/          # User-labeled images
```

## Model Architecture

The CNN consists of:
- 4 convolutional layers (32, 64, 128, 128 filters)
- Max pooling after each conv layer
- 2 dense layers (256, 128 neurons)
- Dropout (0.5, 0.3) for regularization
- Softmax output layer (4 classes)

**Training Configuration:**
- Image size: 192×192 grayscale
- Batch size: 32
- Optimizer: Adam (lr=0.001)
- Data augmentation: horizontal flip, rotation (±15°), zoom (±10%)
- Early stopping on validation loss (patience=5)
- Learning rate reduction on plateau

**Performance:**
- Test Accuracy: ~91%
- Test Recall (Macro): ~90%
- Training time: ~20 epochs
- Model size: ~36 MB

**Note**: Recall is used as the primary metric (prioritizing detection of all tumor cases) alongside accuracy for comprehensive evaluation.

## Dataset

Uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.

**Classes:**
- Glioma: 1,321 training / 300 test images
- Meningioma: 1,339 training / 306 test images
- No Tumor: 1,595 training / 405 test images
- Pituitary: 1,457 training / 300 test images

**Preprocessing:**
1. Convert to grayscale
2. Resize to 192×192
3. Percentile clipping (1st-99th percentile)
4. Min-max normalization to [0, 1]
5. Save as 8-bit PNG

## Makefile Commands

```bash
make env        # Set up Python environment with pyenv
make install    # Install dependencies from requirements.txt
make clean      # Remove cache files and build artifacts
make all        # Run env + install (complete setup)
```

## Troubleshooting

### "No module named 'tensorflow'" or similar import errors
- Ensure you've activated your virtual environment
- Run `pip install -r requirements.txt` again

### "FileNotFoundError: data/brain-tumor-mri-dataset/Testing"
- The dataset is missing - follow the **Dataset Setup** section above
- Verify the folder structure matches exactly as shown

### "OSError: SavedModel file does not exist"
- Models are missing - follow the **Model Files Setup** section above
- Train the model using `python train_improved.py` in `notebooks/exploration/`

### Streamlit app shows "API key not configured"
- This is optional - only needed for AI-assisted report generation
- Create a `.env` file with your Anthropic API key
- You can still use classification features without an API key

### Training takes too long / Out of memory
- Reduce batch size in `train_improved.py` (line ~60: change `batch_size=32` to `batch_size=16`)
- Use a machine with GPU for faster training
- Expect 20-30 minutes with GPU, 2-4 hours with CPU

### Windows: "make: command not found"
- Use the **Manual Setup** instructions instead
- Or install WSL (Windows Subsystem for Linux)

## Requirements

All Python dependencies are listed in `requirements.txt`:

- `streamlit` - Web interface
- `tensorflow` - Deep learning framework
- `opencv-python` - Image processing
- `numpy`, `pandas` - Data manipulation
- `plotly` - Interactive visualizations
- `anthropic` - Claude AI integration
- `python-dotenv` - Environment variable management
- `fpdf` - PDF report generation
- `scikit-learn` - ML utilities
- `jupyter` - Notebook environment
- `fastapi`, `uvicorn` - API framework (future development)

## Future Improvements

- [ ] Expand dataset with user-labeled images (labeling interface already implemented)
- [ ] Add precision and F1-score metrics alongside recall
- [ ] Add batch processing API
- [ ] Deploy as web service (Docker containerization)
- [ ] Support multi-modal MRI sequences (T1, T2, FLAIR)
- [ ] Implement model versioning and A/B testing
- [ ] Add DICOM file format support

## Limitations

- This is an educational project and should not be used for actual medical diagnosis
- Model trained on limited dataset with potential domain shift
- Requires review by qualified medical professionals
- Claude AI reports are generated for demonstration purposes

## Acknowledgments

- Dataset: Masoud Nickparvar ([Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset))
- AI Integration: Anthropic Claude API
- Framework: Streamlit, TensorFlow

## License

For educational purposes only. Dataset license: CC0 1.0 Universal.
