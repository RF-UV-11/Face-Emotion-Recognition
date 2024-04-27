
# 🎭 Face Emotion Recognition

This repository contains a real-time Face Emotion Recognition project using a pre-trained deep learning model to classify facial expressions. The project utilizes computer vision techniques and a trained model to predict emotions (angry, fear, happy, neutral, sad, surprise) from live webcam video feed.

## 🌟 Overview

Facial expression recognition is a technology that allows a computer to recognize human emotions from facial expressions captured by a camera. This project demonstrates real-time emotion recognition using a pre-trained model on a webcam feed.

## 🚀 Usage

### 📋 Requirements

- Python 3
- Jupyter Notebook (`pip install jupyter`)
- OpenCV (`pip install opencv-python`)
- TensorFlow (`pip install tensorflow`)
- Other required libraries (e.g., NumPy, pandas) can be installed using `pip`.

### 🛠️ Setup and Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/RF-UV-11/Face-Emotion-Recognition.git
   cd Face-Emotion-Recognition
   ```

2. **Activate Virtual Environment (Optional):**
   - unzip nenv.zip

   ```bash
   /nenv/Scripts/activate.bat  # Activate virtual environment for Windows
   ```
   ```bash
   source nenv/bin/active # Activate virtual environment Linux
   ```

4. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5.  **Download Data**
   - [Data](https://www.kaggle.com/code/yuvrajjoshi1110/fer-efficientnet/input)
     
6. **Train the Model (Optional):**
   - Open and run the `face_emotion_recognition.ipynb` Jupyter Notebook to train and save the model (`my_model.h5`) for face emotion recognition.

7. **Run the Face Emotion Recognition Script:**

   - Ensure that the pre-trained model (`my_model.h5`) is in the project directory.
   - Execute the following command to start real-time emotion recognition using the webcam:

     ```bash
     python test.py
     ```

     This will activate your webcam and display real-time predictions of facial expressions.

### 🎥 Demo

- Download the demo video from the [`demo`](demo/) folder to see the Face Emotion Recognition system in action.

## 📂 Project Structure

- `face_emotion_recognition.ipynb`: Jupyter Notebook for training and saving the face emotion recognition model.
- `test.py`: Script for real-time face emotion recognition using the trained model.
- `demo/`: Folder containing a demo video showcasing the project in action.

## 🙏 Acknowledgments

This project was inspired by research and implementations in computer vision and deep learning for emotion recognition.

## 👤 Author

- Yuvraj Joshi
- GitHub: [RF-UV-11](https://github.com/RF-UV-11)

## 📝 License

This project is licensed under the MIT License - see the [MIT](https://choosealicense.com/licenses/mit/) file for details.
