Hyperparameter Tuning with Keras Tuner 

This notebook demonstrates hyperparameter tuning using Keras Tuner on the Fashion-MNIST dataset. It compares a baseline CNN model with a hypertuned model and highlights how tuning improves performance and efficiency.

What This Notebook Covers

- Loading and preprocessing the Fashion-MNIST dataset
- Building a baseline CNN classifier
- Defining a hypermodel with tunable parameters
- Using RandomSearch tuner for fast, practical hypertuning
- Selecting the best hyperparameters
- Retraining the best model and evaluating performance

Key Improvements Implemented
- Added a Conv2D layer for better image feature extraction
- Tuned multiple hyperparameters (filters, dense units, dropout, activation, learning rate)
- Saved the best-performing model using ModelCheckpoint

How to Run
- Open the notebook in Google Colab
- Install dependencies:

!pip install -U keras-tuner

