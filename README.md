# Image Classifcation Project

## 1) Description
This prjoect looked at how to create a Image Classifier using a Convolution Neural Network (CNN). Four different networks were investigated, a scratch made on, VGG19, ResNet152V2, and MobileNetV2.

## 2) Installation
All you need to do is download the files and meet the prerequisities below!

### a) Prerequisities
- Python 3+
- Keras & TensorFlow
- Image Dataset from http://vision.stanford.edu/aditya86/ImageNetDogs/

### 3) Usage
Since the images come in a single directory, different directories need to be created for training, validation, and test sets. ProcessImages.py takes the images from the directory, shuffles them, and splits them up into three different directories. Then the four models are created via Scratch.py, VGG19.py, ResNet152V2.py, and MobileNetV2.py. These files train and save each model. Then Predict.py is used to make predictions on the test set.
