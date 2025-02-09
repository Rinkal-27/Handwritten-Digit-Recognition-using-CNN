# MNIST Handwritten Digit Recognition using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels.

## Dataset

The dataset contains 60,000 training images and 10,000 test images.

Each image is labeled as a digit from 0 to 9.

The dataset is preloaded in TensorFlow/Keras and can be imported using:

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

## Dependencies

Make sure you have the following dependencies installed:

pip install tensorflow numpy matplotlib

## Model Architecture

The CNN model consists of the following layers:

Input Layer: 28x28 grayscale images (reshaped to (28,28,1)).

Convolutional Layer 1: 32 filters, kernel size 3x3, ReLU activation.

Max-Pooling Layer 1: Pool size 2x2.

Convolutional Layer 2: 64 filters, kernel size 3x3, ReLU activation.

Max-Pooling Layer 2: Pool size 2x2.

Flatten Layer: Converts 2D matrix into a 1D vector.

Dense (Fully Connected) Layer: 128 neurons, ReLU activation.

Output Layer: 10 neurons (for digits 0-9), Softmax activation.

## Training the Model

Load and preprocess the data

x_train = x_train.reshape(-1, 28, 28, 1) / 255.0

x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

## Compile the model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

## Train the model

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

## Model Evaluation

After training, evaluate the model performance:

loss, accuracy = model.evaluate(x_test, y_test)

print(f'Test Accuracy: {accuracy * 100:.2f}%')

## Results

The model typically achieves around 98% accuracy on the test set.
