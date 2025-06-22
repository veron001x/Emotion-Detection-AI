# Emotion Detection AI 😄😢😠

This project uses a Convolutional Neural Network (CNN) and OpenCV to detect human emotions in real-time using a webcam. It can classify facial expressions into 7 categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

---

## 📁 Project Structure

```
emotion-detection-ai/
├── train_model.py            # Trains the CNN on custom dataset
├── emotion_detector.py       # Runs emotion detection live from webcam
├── requirements.txt          # All required Python packages
├── README.md                 # This file
├── .gitignore                # Files and folders Git should ignore
├── model/
│   └── emotion_model.h5      # Saved model after training (optional)
└── dataset/                  # Not included – download separately
```

---

## 🚀 How to Run

1. Clone this repo and open the folder
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset https://www.kaggle.com/datasets/msambare/fer2013?resource=download and unzip it as `dataset/`
4. Train the model:
   ```bash
   python train_model.py
   ```
5. Run emotion detection live:
   ```bash
   python emotion_detector.py
   ```

Press `Q` to exit the webcam window.

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV (Haar cascades)
- NumPy
- CNN (Convolutional Neural Network)

---

## 📦 Dataset

You can download the dataset used for training https://www.kaggle.com/datasets/msambare/fer2013?resource=download.  
Unzip the folder and place it in the root as:

```
emotion-detection-ai/
└── dataset/
    ├── train/
    └── test/
```

Each of the 7 emotion folders (Angry, Happy, etc.) must be inside both `train/` and `test/`.

---



---

## 🙌 Credits

Thanks to open-source communities and datasets that made this possible.
