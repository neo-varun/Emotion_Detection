# Emotion Detection App

## Overview
This project is an Emotion Detection Application built using PyTorch, Flask, and MediaPipe. It detects facial expressions from images and classifies them into seven emotional states: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The application uses a CNN model trained on the FER2013 dataset, enhanced with MediaPipe face detection for improved accuracy.

## Features
- **Smart Emotion Detection** – Upload images and get instant emotion predictions
- **Advanced Face Detection** – Uses MediaPipe for accurate face detection and cropping
- **Interactive Web Interface** – User-friendly UI with visualization
- **Model Training Interface** – Train the model directly through the web interface
- **Comprehensive Analytics** – View detailed model performance metrics
- **CUDA Support** – GPU acceleration for faster training and inference

## Prerequisites

### Dataset Download
Before running the application, you need to download the FER2013 dataset from Kaggle:
1. Visit [FER2013 Dataset on Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
2. Download and extract the dataset
3. Place the extracted images in the following structure:
```
data/
└── fer2013/
    ├── train/
    │   ├── 0/
    │   ├── 1/
    │   └── ...
    ├── val/
    │   ├── 0/
    │   └── ...
    └── test/
        ├── 0/
        └── ...
```

### System Requirements
- Python 3.8+
- CUDA-capable GPU (optional but recommended for training)

## Installation & Setup

### Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application
1. Activate your virtual environment
2. Launch the Flask server:
```bash
python app.py
```
3. Open your web browser and navigate to: `http://localhost:8000`

## How the Program Works

### Application Components
1. **Face Detection (data_preprocessing.py)**
   - Uses MediaPipe for accurate face detection
   - Preprocesses images for consistent input size (48x48)
   - Supports batch processing for dataset preparation

2. **Model Architecture (dataset_training.py)**
   - CNN-based architecture optimized for emotion detection
   - Includes batch normalization and dropout for regularization
   - Supports CUDA acceleration for faster training

3. **Training Pipeline**
   - Automated dataset loading and preprocessing
   - Mixed-precision training for improved performance
   - Automatic model checkpointing
   - Detailed performance metrics tracking

4. **Web Interface**
   - Upload images for emotion detection
   - View detection confidence scores
   - Train model through the interface
   - Monitor training progress and metrics

### Performance Metrics & Evaluation
The model's performance is evaluated using multiple metrics:

1. **Overall Metrics**
   - Accuracy: Overall prediction accuracy across all emotions
   - Precision: Ability to identify true positives accurately
   - Recall: Ability to find all relevant instances
   - F1 Score: Harmonic mean of precision and recall

### Usage Guide

1. **Dataset Preparation**
   - Download and extract the FER2013 dataset
   - Place it in the correct directory structure
   - Verify data organization before training

2. **Model Training**
   - Click "Start Training" on the web interface
   - Monitor training progress and metrics
   - View detailed performance metrics after training
   - Check convergence and validation metrics

3. **Emotion Detection**
   - Upload an image through the web interface
   - View detected emotion and confidence scores
   - Review face detection accuracy

## Technologies Used
- **PyTorch** (Deep Learning)
- **Flask** (Web Server)
- **MediaPipe** (Face Detection)
- **OpenCV** (Image Processing)
- **NumPy** (Numerical Operations)
- **scikit-learn** (Metrics Calculation)

## Performance Metrics
The model is evaluated on multiple metrics:
- Accuracy score
- Precision score
- Recall score
- F1 score

## License
This project is licensed under the MIT License.

## Author
Developed by Varun. Feel free to connect with me:
- Email: darklususnaturae@gmail.com
