# GTSRB Traffic Sign Classification using PyTorch

## Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model is trained to recognize 43 different traffic sign categories and achieves approximately 92% test accuracy.

The project covers the complete deep learning workflow, including data preprocessing, data augmentation, model training, validation, testing, and performance evaluation using confusion matrices.

---

## Dataset

**Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)

* 43 traffic sign classes
* More than 39,000 training images
* Official test set with labeled annotations
* Images contain varying lighting conditions, scales, and viewing angles

Dataset Source:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

---

## Project Objectives

* Build a custom CNN architecture using PyTorch
* Train the model to classify traffic signs into 43 categories
* Apply image augmentation techniques to improve generalization
* Evaluate model performance on validation and test datasets
* Analyze model predictions using confusion matrices

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Google Colab

---

## Model Architecture

The CNN consists of:

### Feature Extraction Layers

1. Convolution Layer (3 → 32)

2. ReLU Activation

3. Max Pooling

4. Convolution Layer (32 → 64)

5. ReLU Activation

6. Max Pooling

7. Convolution Layer (64 → 128)

8. ReLU Activation

9. Max Pooling

### Classification Layers

1. Fully Connected Layer (8192 → 512)
2. ReLU Activation
3. Dropout (0.3)
4. Output Layer (512 → 43)

---

## Data Preprocessing

Training images are processed using:

* Resize to 64×64 pixels
* Random Rotation
* Color Jitter (Brightness and Contrast)
* Conversion to Tensor

Validation and test images are only resized and converted to tensors.

---

## Training Configuration

* Optimizer: Adam
* Learning Rate: 0.001
* Loss Function: CrossEntropyLoss
* Batch Size: 32
* Epochs: 10

---

## Results

| Metric              | Score |
| ------------------- | ----- |
| Validation Accuracy | ~92%  |
| Test Accuracy       | ~92%  |

The model successfully learns meaningful traffic sign features and performs well on unseen test data.

---

## Confusion Matrix Analysis

A confusion matrix was generated to evaluate class-wise performance.

Observations:

* Most predictions lie along the main diagonal, indicating correct classifications.
* Misclassifications occur primarily between visually similar traffic signs.
* The model demonstrates strong generalization across the majority of classes.

---

## Project Structure

```text
GTSRBTrafficSignClassifier/
│
├── GTSRB_Classification.ipynb
├── README.md
├── confusion_matrix.png
└── requirements.txt
```

---

## Future Improvements

Potential improvements include:

* Transfer Learning using ResNet18 or EfficientNet
* Hyperparameter tuning
* Early stopping
* Learning rate scheduling
* Streamlit web application for real-time traffic sign prediction
* Model deployment using Docker and FastAPI

---

## Key Skills Demonstrated

* Deep Learning
* Computer Vision
* Image Classification
* PyTorch
* Data Augmentation
* Model Evaluation
* Confusion Matrix Analysis
* Custom Dataset Creation
* Neural Network Training

---

## Conclusion

This project demonstrates the end-to-end development of a computer vision classification system using PyTorch. The trained CNN achieves approximately 92% accuracy on the GTSRB dataset and provides a solid foundation for more advanced traffic sign recognition and autonomous driving applications.

