# EmoDetect – Real-Time Emotion Recognition

A CNN-powered webcam system that detects and classifies **7 human emotions in real time** using a custom VGG-inspired architecture trained on the FER2013 dataset.


## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results](#results)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)

## Demo

Run `webcam_demo()` in `Emotion_App.ipynb` to launch the live webcam feed. The system draws a color-coded bounding box around each detected face and overlays the predicted emotion with a confidence percentage. 
Or test on a single image:
```python
predict_single_image(r"path/to/your/image.jpg")
```
NB: Make sure you have the FER2013 folder dowloaded and change the paths to all the files/folders present in the code ('Emotion_App.ipynb)

## Features

- **Real-time face detection** using OpenCV Haar Cascade
- **7-class emotion classification** (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised)
- **Temporal smoothing** — predictions are averaged over the last 8 frames to eliminate label flickering
- **Class-weighted training** — Disgust (only 436 samples) gets 9.4× more attention during training
- **CPU-only inference** — no GPU required, runs on any standard laptop
- **Two-notebook structure** — clean separation between training and inference


## Project Structure

```
EmoDetect/
│
├── Model_Training.ipynb          # Data loading, augmentation, CNN training, evaluation
├── Emotion_App.ipynb             # Load weights, webcam demo, single-image inference
│
├── FER2013_results/              # Created automatically during training
│   ├── emotion_model.weights.h5  # Saved model weights (~21.8 MB)
│   ├── training_curves.png       # Accuracy & loss plots
│   └── confusion_matrix.png      # Per-class confusion matrix
│
└── README.md
```

> **Note:** The FER2013 dataset and trained weights are **not** included in this repo due to size. See [Getting Started](#getting-started) for setup instructions.

---

## Dataset

**FER2013** — 35,887 grayscale 48×48 pixel facial images across 7 emotion classes.

| Emotion   | Training Samples | Class Weight |
|-----------|-----------------|--------------|
| Happy     | 8,989           | 1.0×         |
| Neutral   | 6,198           | 1.3×         |
| Sad       | 6,077           | 1.3×         |
| Fear      | 5,121           | 1.6×         |
| Angry     | 4,953           | 1.6×         |
| Surprise  | 4,002           | 2.0×         |
| **Disgust** | **436**       | **9.4×**     |

The severe class imbalance (Disgust is 20× smaller than Happy) is addressed through computed class weights and heavy data augmentation.

Download: [Kaggle – FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

Expected folder structure after extracting:
```
FER2013/
├── train/
│   ├── angry/
│   ├── disgusted/
│   ├── fearful/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
└── test/
    └── (same structure)
```


## Model Architecture

A VGG-inspired CNN with **1,899,751 total parameters (7.25 MB)**. Built for 48×48 grayscale input.

```
Input (48 × 48 × 1)
│
├── Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
├── Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
├── Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
├── Block 4: Conv2D(256) → BN → MaxPool → Dropout(0.25)
│
├── Flatten
├── Dense(512) → BN → Dropout(0.4)
├── Dense(256) → Dropout(0.3)
└── Dense(7, softmax)
```

BatchNormalization after every Conv layer stabilizes training on the small 48×48 input. Progressive dropout (0.25 → 0.4 → 0.3) prevents overfitting without sacrificing capacity.

---

## Getting Started

### Prerequisites

- Python 3.9+
- Anaconda recommended (the notebooks were developed in a conda base environment)

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/emodetect.git
cd emodetect

# Install dependencies
pip install tensorflow opencv-python scikit-learn matplotlib seaborn ipywidgets
```

Or install all at once via the first cell in either notebook:
```bash
pip install tensorflow opencv-python matplotlib ipywidgets
```

### Path Configuration

Before running, update the path variables in each notebook:

**`Model_Training.ipynb` — cell 2:**
```python
DATASET_ROOT = r"C:\Users\YourName\Desktop\FER2013"   # path to FER2013 folder
SAVE_DIR     = r"C:\Users\YourName\Desktop\FER2013_results"
```

**`Emotion_App.ipynb` — cell 3:**
```python
WEIGHTS_PATH = r"C:\Users\YourName\Desktop\FER2013_results\emotion_model.weights.h5"
```


## Usage

### Step 1 — Train the model

Open and run all cells in **`Model_Training.ipynb`**. This will:
1. Load FER2013 and compute class weights
2. Apply data augmentation (rotation, zoom, horizontal flip)
3. Train for up to 50 epochs with early stopping
4. Save weights to `emotion_model.weights.h5`
5. Generate training curves and a confusion matrix

Training time is roughly 2–4 hours on CPU (much faster with GPU).

### Step 2 — Run the app

Open **`Emotion_App.ipynb`** and run all cells. The notebook will:
1. Rebuild the exact model architecture
2. Load the saved weights
3. Start the webcam demo (or run on a static image)

**Live webcam:**
```python
webcam_demo()    # press 'q' to quit
```

**Single image:**
```python
predict_single_image(r"path/to/image.jpg")
```

---

## Training Details

| Hyperparameter     | Value                              |
|--------------------|------------------------------------|
| Optimizer          | Adam (lr = 0.001)                  |
| Epochs             | Up to 50                           |
| Batch size         | 64                                 |
| LR schedule        | ReduceLROnPlateau (factor=0.5)     |
| Early stopping     | Patience = 10 epochs               |
| Checkpoint         | Best validation accuracy           |
| Augmentation       | Rotation ±15°, zoom 10%, h-flip    |
| Class weights      | sklearn `compute_class_weight`     |

**Temporal smoothing:** The app maintains a `deque(maxlen=8)` of raw predictions. The displayed label is the `argmax` of the mean across those 8 frames — this prevents the label from flickering every frame.

---

## Results

~63% validation accuracy. It is considered strong for FER2013, which is a notoriously noisy dataset. Fear/Sad confusion is expected and mirrors human inter-rater disagreement on this dataset.

---

## Known Limitations

- **Lighting sensitivity** — performance drops significantly in poor lighting. A bright, front-facing light source helps.
- **Single-face optimized** — the temporal buffer is shared globally, so results may be inconsistent when multiple faces are in frame simultaneously.
- **FER2013 bias** — the dataset was collected from web searches, so it skews toward exaggerated or posed expressions rather than subtle real-world emotions.
- **No GPU on native Windows (TF ≥ 2.11)** — use WSL2 or the TensorFlow-DirectML plugin for GPU acceleration on Windows.




---

## License

This project is for educational purposes. FER2013 is publicly available on Kaggle under its own terms of use.
