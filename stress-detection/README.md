# Face-Expression-Recognition-using-Deep-Learning
This project implements a convolutional neural network (CNN) to recognize facial expressions of seven different emotions: angry, disgust, fear, happy, neutral, sad, and surprise. The model is trained on the Face expression recognition dataset. Dataset E-link: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset.

**NEW FEATURE**: Real-time facial feature tracking including eyes, eyebrows, nose, and mouth detection!

## Features:
- 🎭 **Emotion Recognition**: Detects 7 different emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- 👁️ **Eye Detection**: Tracks left and right eyes in real-time
- 🤨 **Eyebrow Tracking**: Monitors both left and right eyebrows
- 👃 **Nose Detection**: Identifies and tracks the nose
- 👄 **Mouth Tracking**: Detects mouth position and shape
- 🎨 **Visual Overlay**: Color-coded facial landmarks (Blue=Eyes, Green=Eyebrows, Red=Nose, Yellow=Mouth)

## Requirements:
```
Python 3.11
keras>=2.13.0
tensorflow>=2.13.0
numpy>=1.23.5
matplotlib>=3.7.0
pandas>=1.5.3
seaborn>=0.12.0
opencv-contrib-python>=4.8.0
dlib>=19.24.0
Flask
```

## Installation:
1. Install Python 3.11 or higher
2. Clone or download the repository
3. Install packages: `pip install -r requirements.txt`
4. Download the dlib facial landmarks model:
   - Download `shape_predictor_68_face_landmarks.dat` from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - Extract and place it in the project root directory

## Usage:

### Option 1: Web Interface (Recommended)
```bash
python app.py
```
Then open your browser and navigate to: `http://localhost:5000`

### Option 2: Command Line
```bash
python main.py
```

## Files:
- **app.py**: Flask web application for real-time emotion and facial feature detection
- **main.py**: Command-line entry point for the emotion detection
- **emotion_recognition_cnn.py**: CNN model building and training code
- **HaarcascadeclassifierCascadeClassifier.xml**: Pre-trained Haar Cascade Classifier for face detection
- **model.h5**: Pre-trained Keras model for emotion detection
- **templates/index.html**: Web interface with real-time emotion and facial feature display
- **download_models.py**: Helper script for downloading required models

## Facial Features Information:
The application tracks 68 facial landmarks using the dlib face detection library:
- **Eyes (Blue)**: Landmarks 36-47 (eyes region)
- **Eyebrows (Green)**: Landmarks 17-26 (eyebrow region)
- **Nose (Red)**: Landmarks 27-35 (nose region)
- **Mouth (Yellow)**: Landmarks 48-67 (mouth region)

These landmarks are displayed as colored circles overlaid on the detected face in the video feed.

## Troubleshooting:
If you see a warning about `shape_predictor_68_face_landmarks.dat` not found:
1. The facial landmarks tracking will be disabled
2. Download the file from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
3. Extract it and place in the project root directory
4. Restart the application
