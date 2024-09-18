# Emotion detection using deep learning

> [!CAUTION]
> This repository is licensed with a modified Apache 2.0 License which adds a non-commerical use clause

## Introduction

This project implements a real-time emotion detection system using OpenCV for face and eye detection, and a pre-trained convolutional neural network (CNN) model for emotion recognition. The system captures video feed from a webcam, detects faces and eyes using Haar cascades, and classifies emotions such as 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', and 'Neutral' using a Keras model trained with FER-2013 Dataset. The system visualizes both the face bounding box and predicted emotion in real time.
## Dependencies

* OpenCV
* TensorFlow
* Keras
* NumPy
* Pandas

To install the required packages, run `pip install -r requirements.txt`.

## Basic Information

The directory structure is of the form: 

```md
RealTime-Emotion-Detector
├── config
│   ├── frame_drawing.py
│   └── frame_editing.py
├── config
│   └── fer2013.csv.py
├── model
│   └── emotion_model.h5
├── data_preprocessing.py
├── emotion_detector.py
├── model_architecture.py
└── train_emotion_model.json
```
---
![Accuracy plot](https://githubreadme.s3.eu-north-1.amazonaws.com/trainGraph.png)

With a simple 4-layer CNN, the test accuracy reached 60% in 50 epochs showing how validation accuracy slightly outperforming the training accuracy. This suggests that the model generalizes well to unseen data, without significant overfitting.

---


## Basic Usage

To run the live emotion detection system with the included model, follow these steps:

0. Clone this repository and navigate to the directory
```bash
git clone https://github.com/davidpm-19/RealTime-Emotion-Detector.git
cd Emotion-detection
```

1. Run the live detection script: Start the live webcam feed by running the LiveFaceDetection class, which will automatically load the Haar cascades for face and eye detection, and the emotion recognition model.
```bash
python emotion_detector.py
```

2. Usage in real-time: Once the webcam feed starts, the system will:

* Detect faces and eyes in the video feed.
* Predict the emotion for each detected face.
* Display the predicted emotion on the video feed in real-time.

3. Quit: Press `q` at any time to stop the live webcam feed and close the application.


## Data Preparation (optional)

* The [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset is available to download but I also insided it inside `dataset` subdirectory

* In case you are looking to experiment with new datasets you can access `data_preprocessing.py` which is ready to handle csv files and can be used as a reference.

## Model Architecture (optional)

The emotion detection model is built using a simple convolutional neural network with 4 layers.<br>
If you want to try the included model before experimenting with the code here's an overview of the architecture:

1. **Input Layer:** Takes 48x48 grayscale images.
2. **Convolutional Layers:** 
    - 4 convolutional layers are used, progressively increasing the number of filters (32, 64, 128, 128).
    - Max-pooling layers follow each convolutional layer to downsample the feature maps.
3. **Dropout Layers:**
    - Dropout of 25% after the convolutional layers for regularization.
    - Dropout of 50% after the fully connected layer to prevent overfitting.
4. **Fully Connected Layer:**
    - A fully connected layer with 1024 neurons and ReLU activation.
5. **Output Layer:**
    - A dense layer with 7 units (corresponding to 7 emotion classes) and softmax activation for multiclass classification.

The model is compiled using the Adam optimizer and categorical cross-entropy as the loss function. It tracks accuracy as the primary metric.

## Train Emotion Model (optional)

1. **Data Augmentation:**
    - ImageDataGenerator is used to perform real-time data augmentation, including random rotations, width and height shifts, zooming, and horizontal flips. This improves the model's robustness and helps prevent overfitting.
    
2. **Training:**
    - The model is trained for 50 epochs using the augmented training data, with a validation split to track performance on unseen data. The batch size is set to 64.

3. **Model Saving:**
    - After training, the trained model is saved in the `model` directory as `emotion_model.h5`.

## Algorithm

* First, the **Haar cascade** method is used to detect faces in each frame of the webcam feed using the `haarcascade_frontalface_default.xml` file.
  
* If faces are detected, **eye detection** is performed using the `haarcascade_eye.xml` file to further enhance accuracy.
  
* The region of the image containing the face is converted to grayscale and resized to **48x48**, as required by the pre-trained emotion recognition model.

* The resized face region is passed to the **CNN** model, which outputs a list of **softmax scores** corresponding to seven emotion classes.

* The emotion with the maximum score is selected, and its label is displayed on the screen along with a white rectangle drawn around the detected face.

* The model continuously processes frames from the webcam feed, detecting faces, identifying emotions, and displaying the results in real-time.
