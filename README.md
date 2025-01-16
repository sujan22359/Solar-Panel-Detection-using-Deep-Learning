# Solar Panel Detection using CNN

## Overview
This project aims to detect solar panels in images using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. The model is trained on a dataset of labeled images and can accurately identify the presence of solar panels.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sujan22359/Solar-Panel-Detection-using-Deep-Learning.git
## Dataset
The dataset consists of images of various areas with and without solar panels. Each image is labeled with 0 (no solar panel) or 1 (solar panel).
## Model Architecture
The model is a Convolutional Neural Network (CNN) with the following architecture:
Multiple Conv2D layers with ReLU activation and BatchNormalization
MaxPooling2D layers to reduce spatial dimensions
GlobalMaxPooling2D layer for feature extraction
Dense layer with sigmoid activation for binary classification
## Training
The model is trained using the binary cross-entropy loss function and the Adam optimizer. The training process involves data augmentation and stratified k-fold cross-validation to ensure robust performance.
## Evaluation
The model's performance is evaluated using accuracy, ROC-AUC score, and confusion matrix. The evaluation results show the model's ability to accurately detect solar panels in images.
## Usage
To use the trained model for prediction:

   # Load the model:

     from tensorflow.keras.models import load_model
     model = load_model('solar_panel_detection_model.h5')
   # Make predictions on new images:

     import cv2
     import numpy as np

     def predict(image_path, model):
       img = cv2.imread(image_path)
       img = cv2.resize(img, (101, 101))
       img = np.expand_dims(img, axis=0)
       img = img / 255.0
       prediction = model.predict(img)
       return 'Solar Panel Detected' if prediction[0][0] > 0.5 else 'No Solar Panel Detected'

     result = predict('path_to_image.jpg', model)
     print(result)
## Results
The model achieved high accuracy and ROC-AUC score on the test set, demonstrating its effectiveness in detecting solar panels. Below are some sample predictions:


