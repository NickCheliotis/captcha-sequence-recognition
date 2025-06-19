# CAPTCHA Recognition with CNN + RNN + CTC
This project tackles CAPTCHA image decoding using a hybrid architecture combining Convolutional Neural Networks (CNNs) and Bidirectional LSTMs, trained using Connectionist Temporal Classification (CTC) loss.
The goal is to decode 5-character CAPTCHA strings from distorted images without needing character segmentation.

## Architecture
CNN: Extracts spatial features from input images

BiLSTM: Captures sequence dependencies along image width.

CTC Loss: Enables training without needing aligned character labels.

## Results
Final accuracy: ~53%

Training duration: 50 epochs

Batch size: 64

##  Files

DataPreprocessing.py -> Used to preprocess the images.

*model.ipnyb -> Used to train/evaluate the model

*To run this notebook, first upload your dataset and any necessary files to your Google Drive, then mount it within the Colab environment.
