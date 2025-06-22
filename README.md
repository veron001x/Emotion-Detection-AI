# Emotion Detection from Facial Expressions

This project uses a Convolutional Neural Network (CNN) to detect human emotions from facial expressions in real-time using a webcam.

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download the FER-2013 dataset and place it inside the `dataset/` folder.

3. Train the model:
   ```
   python train_model.py
   ```

4. Run emotion detection:
   ```
   python emotion_detector.py
   ```

## Emotion Classes
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral