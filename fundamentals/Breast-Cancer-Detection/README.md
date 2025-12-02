# Breast Cancer Detection using Deep Learning


## Project Overview
This project implements a deep learning solution for breast cancer detection using Convolutional Neural Networks(CNN). The model is trained on a dataset of medical images(breast tissue biopsies) to classify between benign and malignant cases, achieving high accuracy in detection.


## Features
- **Deep Learning Model**: Custom CNN architecture optimized for medical image classification
- **Data Augmentation**: Comprehensive augmentation pipeline including:
  - Random horizontal and vertical flips
  - Rotation(±10%)
  - Zoom (±10%)
  - Contrast Adjustments
  - Translation
- **Training Pipeline**:
  - 70/20/10 train/validation/test split
  - Early stopping implementation
  - Learning rate reduction on plateau
  - TensorBoard integration for monitoring

## 🛠️ Technical Stack
- Python 3.12.3
- TensorFlow 2.19.0
- Matplotlib 3.10.3

## 🚀 Getting Started

### Prerequisites
- Python >= 3.12
- pip

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/breast_cancer_detection.git
cd breast_cancer_detection
```

2. Create and activate virtual environment:
```bash
python- m venv venv
source venv/bin/activate # For Mac + WSL
```
3. Install dependencies:
```bash
pip install tensorflow matplotlib
```

## Usage
1. Place your dataset in the '/data' directory
2. Run the training script:
```bash
python main.py
```

## 📊 Model Architecture
The model implements a CNN with the following structure:
- Input Layer: 256x256x3 images
- Convolutional Layers with Batch Normalization
- MaxPooling and Dropout for regularization
- Dense layers for classification
- Binary output (benign/malignant)

## 📈 Results
The model achieves:
- Training accuracy: ~87%
- Validation accuracy: ~88%

## Future Improvements
- [ ] Add model interpretation capabilities
- [ ] Implement additional evaluation metrics
- [ ] Create a simple API for predictions
