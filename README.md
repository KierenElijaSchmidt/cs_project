# NeuroSight - Brain Tumor MRI Classification

A deep learning application for classifying brain MRI scans into four categories: glioma, meningioma, pituitary tumor, and no tumor. Built with TensorFlow and Streamlit.

## Features

- **CNN-based Classification**: Custom convolutional neural network achieving 90%+ test accuracy
- **Web Interface**: Interactive Streamlit dashboard for uploading and classifying MRI scans
- **Model Performance Analysis**: Test the model on random samples with detailed accuracy metrics
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

- Python 3.8.12 (managed via pyenv)
- pip
- Git

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

### Environment Configuration

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

Get your API key from [Anthropic Console](https://console.anthropic.com/).

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
- Generate AI-assisted medical reports
- Save labeled images to improve the model

**2. Test Model Performance**
- Select random images from the test dataset
- View predictions vs actual labels
- Analyze per-class accuracy and overall performance
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
- Training time: ~20 epochs
- Model size: ~36 MB

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

- [ ] Add confusion matrix visualization
- [ ] Implement grad-CAM for interpretability
- [ ] Expand dataset with user-labeled images
- [ ] Add batch processing API
- [ ] Deploy as web service
- [ ] Support multi-modal MRI sequences (T1, T2, FLAIR)

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
