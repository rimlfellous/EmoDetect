 
# EmoDetect
A CNN-powered webcam system that detects and classifies 7 human emotions in real time, trained on FER2013.

##  Project Structure
```
EmoDetect/
│
├── Model_Training.ipynb          # Training pipeline
├── Emotion_App.ipynb             # Webcam demo & single-image inference
├── app.py                        # Standalone terminal app
│
├── FER2013_results/
│   ├── emotion_model.weights.h5
│   ├── training_curves.png
│   └── confusion_matrix.png
│
└── README.md
```
> The FER2013 dataset and trained weights are not included due to size.

##  Features
- Real-time face detection via OpenCV Haar Cascade
- 7-class emotion classification (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised)
- Temporal smoothing over 8 frames — no flickering labels
- Class-weighted training — Disgust gets 9.4× more attention (only 436 samples)
- CPU-only — no GPU needed, runs on any laptop
- Standalone `app.py` — run the demo straight from terminal, no Jupyter needed

##  Getting Started
**Prerequisites:** Python 3.9+, Anaconda recommended
```bash
git clone https://github.com/your-username/emodetect.git
cd emodetect
pip install tensorflow opencv-python scikit-learn matplotlib seaborn ipywidgets
```

### Path Configuration
Update these before running anything:

**`Model_Training.ipynb` — cell 2:**
```python
DATASET_ROOT = r"C:\Users\YourName\Desktop\FER2013"
SAVE_DIR     = r"C:\Users\YourName\Desktop\FER2013_results"
```

**`Emotion_App.ipynb` — cell 3:**
```python
WEIGHTS_PATH = r"C:\Users\YourName\Desktop\FER2013_results\emotion_model.weights.h5"
```

##  Usage

### Option A — Jupyter
1. Run all cells in `Model_Training.ipynb` to train *(2–4 hrs on CPU)*
2. Run all cells in `Emotion_App.ipynb` to launch the demo
```python
webcam_demo()                               # press Q to quit
predict_single_image(r"path/to/image.jpg")  # or test a single image
```

### Option B — Terminal (app.py)
1. Open `app.py` and set your weights path:
```python
WEIGHTS_PATH = r"C:\Users\YourName\Desktop\FER2013_results\emotion_model.weights.h5"
```

2. Run:
```bash
python app.py                         # webcam demo
python app.py --image path/to/image   # single image
```

Press **Q** to quit.

> **Troubleshooting:** If the webcam window doesn't open, check camera permissions. If weights fail to load, verify `WEIGHTS_PATH` is correct. If TensorFlow errors appear, confirm you're in the right environment.

##  Dataset
FER2013 — 35,887 grayscale 48×48 images, 7 classes. [Download on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

| Emotion | Samples | Class Weight |
|---------|---------|--------------|
| Happy | 8,989 | 1.0× |
| Neutral | 6,198 | 1.3× |
| Sad | 6,077 | 1.3× |
| Fear | 5,121 | 1.6× |
| Angry | 4,953 | 1.6× |
| Surprise | 4,002 | 2.0× |
| Disgust | 436 | 9.4× |

##  Model Architecture
VGG-inspired CNN — 1,899,751 params (7.25 MB), built for 48×48 grayscale input.
```
Input (48 × 48 × 1)
├── Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
├── Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
├── Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
├── Block 4: Conv2D(256) → BN → MaxPool → Dropout(0.25)
├── Flatten
├── Dense(512) → BN → Dropout(0.4)
├── Dense(256) → Dropout(0.3)
└── Dense(7, softmax)
```

##  Training Details
| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (lr = 0.001) |
| Epochs | Up to 50 |
| Batch size | 64 |
| LR schedule | ReduceLROnPlateau (factor=0.5) |
| Early stopping | Patience = 10 epochs |
| Augmentation | Rotation ±15°, zoom 10%, h-flip |

##  Results
~63% validation accuracy on FER2013. Fear/Sad confusion is expected and mirrors human inter-rater disagreement on this dataset.

##  Known Limitations
- Performance drops in poor lighting
- Temporal buffer is shared globally — inconsistent with multiple faces in frame
- Dataset skews toward posed/exaggerated expressions, not subtle real-world ones
- No GPU support on native Windows (TF ≥ 2.11) — use WSL2 or TensorFlow-DirectML

##  License
Educational use only. FER2013 is publicly available on Kaggle under its own terms of use.
