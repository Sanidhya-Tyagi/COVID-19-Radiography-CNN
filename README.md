# COVID-19 X-Ray Detection using a Custom CNN (PyTorch)

A deep learning project that trains a custom Convolutional Neural Network from scratch to classify chest X-ray images into four categories: **COVID-19**, **Normal**, **Lung Opacity**, and **Viral Pneumonia**.

This project does not use any pre-trained models or transfer learning. The entire CNN architecture is designed and trained from the ground up on the COVID-19 Radiography Database.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Usage](#usage)
- [Graphs and Visualisations](#graphs-and-visualisations)
- [Single Image Inference](#single-image-inference)
- [Limitations and Disclaimer](#limitations-and-disclaimer)

---

## Overview

Chest X-rays are one of the primary tools used by clinicians to assess lung conditions. Automating the classification of these images using deep learning can assist in faster screening, especially in resource-limited settings.

This project builds a CNN classifier entirely from scratch using PyTorch. The model learns to distinguish between four lung conditions directly from raw pixel data, without relying on weights from models pre-trained on ImageNet or any other dataset.

Key highlights:
- Custom CNN architecture with 4 convolutional blocks
- Batch normalisation and dropout for regularisation
- Data augmentation to improve generalisation
- Stratified train / val / test split
- Full evaluation suite: confusion matrix, ROC curves, per-class metrics, confidence plots
- Single image inference with a probability breakdown

---

## Dataset

**COVID-19 Radiography Database**  
Source: [Kaggle — tawsifurrahman](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

The dataset contains chest X-ray images in PNG format across four classes:

| Class            | Description                                  |
|------------------|----------------------------------------------|
| COVID            | X-rays from confirmed COVID-19 patients      |
| Normal           | X-rays from healthy individuals              |
| Lung Opacity     | Non-COVID lung opacity cases                 |
| Viral Pneumonia  | X-rays from viral pneumonia patients         |

The dataset contains over 21,000 images in total. Each image is a grayscale chest X-ray at 299x299 pixels, which is resized to 128x128 for training.

To download the dataset using the Kaggle API:

```bash
kaggle datasets download -d tawsifurrahman/covid19-radiography-database --unzip -p ./data
```

Expected folder structure after extraction:

```
data/
└── COVID-19_Radiography_Dataset/
    ├── COVID/
    │   └── images/
    ├── Normal/
    │   └── images/
    ├── Lung_Opacity/
    │   └── images/
    └── Viral Pneumonia/
        └── images/
```

---


## Model Architecture

The model is a fully custom CNN with no borrowed weights. It follows a classic encoder-style design where spatial resolution decreases and feature depth increases as data passes through the network.

```
Input: (B, 3, 128, 128)

ConvBlock 1:  Conv2d(3, 32)   -> BN -> ReLU -> Conv2d(32, 32)   -> BN -> ReLU -> MaxPool2d(2)
ConvBlock 2:  Conv2d(32, 64)  -> BN -> ReLU -> Conv2d(64, 64)   -> BN -> ReLU -> MaxPool2d(2)
ConvBlock 3:  Conv2d(64, 128) -> BN -> ReLU -> Conv2d(128, 128) -> BN -> ReLU -> MaxPool2d(2)
ConvBlock 4:  Conv2d(128,256) -> BN -> ReLU -> Conv2d(256, 256) -> BN -> ReLU -> MaxPool2d(2)

Global Average Pooling  ->  (B, 256)

FC1:  Linear(256, 256) -> ReLU -> Dropout(0.5)
FC2:  Linear(256, 4)

Output: (B, 4) logits
```

**Design decisions:**
- **Double conv per block** — two conv layers before each pool gives the network a larger effective receptive field without increasing depth excessively
- **Batch Normalisation** — applied after every conv layer; stabilises training and acts as a mild regulariser
- **Global Average Pooling** — replaces a large flatten+FC combination, significantly reducing parameter count and overfitting risk
- **Dropout (0.5)** — applied before the final classification layer to further reduce overfitting
- **Kaiming initialisation** — all conv and linear layers are initialised with He normal init, which is well suited for ReLU activations

---

## Results

> Results will vary depending on hardware, random seed, and number of epochs.  
run.
> Current results give a 93% Test Accuracy with 98+ ROC.
---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Sanidhya-Tyagi/COVID-19-Radiography-CNN.git
cd COVID-19-Radiography-CNN
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
Check Requirements Section
```

### 4. Download the dataset

Place your `kaggle.json` API token in `~/.kaggle/`, then run:

```bash
kaggle datasets download -d tawsifurrahman/covid19-radiography-database --unzip -p ./data
```

### 5. Open and run the notebook

```bash
jupyter notebook main.ipynb
```

Run all cells from top to bottom. Training will save `best_model.pt` automatically.

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy
matplotlib
seaborn
scikit-learn
Pillow
tqdm
jupyter
kaggle
```

Install all at once:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn Pillow tqdm jupyter kaggle
```

---

## Usage

### Training

Open `main.ipynb` and run all cells. Key parameters you can adjust at the top of the notebook:

| Parameter   | Default | Description                          |
|-------------|---------|--------------------------------------|
| `IMG_SIZE`  | 128     | Input image size (pixels)            |
| `BATCH_SIZE`| 32      | Training batch size                  |
| `NUM_EPOCHS`| 20      | Number of training epochs            |
| `LR`        | 1e-3    | Initial learning rate                |

For faster runs on CPU, reduce `IMG_SIZE` to 64 and `NUM_EPOCHS` to 5.

### Loading a saved model

```python
model = CustomCNN(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load('best_model.pt', map_location=DEVICE))
model.eval()
```

---

## Graphs and Visualisations

After training, run the graphs code block to generate the following plots:

| Graph                          | Description                                              |
|--------------------------------|----------------------------------------------------------|
| Training curves                | Loss and accuracy for train and val over all epochs      |
| Confusion matrix (counts)      | Raw prediction counts per class                          |
| Confusion matrix (normalised)  | Per-class accuracy as a percentage                       |
| Per-class metrics              | Precision, recall, and F1 for each class side by side    |
| Class distribution             | Number of test images per class                          |
| ROC curves                     | One-vs-rest ROC with AUC for each class                  |
| Confidence distribution        | How confident the model is on correctly labelled samples |
| Correct vs wrong predictions   | Bar chart comparing hits and misses per class            |

All plots are saved as PNG files in the working directory.

---

## Single Image Inference

To classify a single chest X-ray image:

```python
IMAGE_PATH = 'path/to/your/xray.png'
```

The inference block will:
1. Load and preprocess the image
2. Run it through the trained model
3. Print the predicted class and confidence score to the console
4. Display a side-by-side plot of the X-ray and a horizontal bar chart showing the probability assigned to each class

---

## Limitations and Disclaimer

**This project is for educational and research purposes only. It is not a medical diagnostic tool and must not be used for clinical decision-making.**

- The model is trained on a single publicly available dataset and has not been validated on external clinical data
- X-ray classification is a complex task influenced by imaging equipment, patient positioning, and radiologist annotations
- Class imbalance in the dataset may affect per-class performance
- COVID-19 diagnosis requires PCR testing and clinical evaluation by a qualified medical professional

---

## Acknowledgements

- Dataset: M.E.H. Chowdhury et al., *Can AI Help in Screening Viral and COVID-19 Pneumonia?*, IEEE Access, 2020
- PyTorch team for the deep learning framework
- Kaggle for hosting the dataset
