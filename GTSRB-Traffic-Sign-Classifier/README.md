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
| Validation Accuracy | ~99%  |
| Test Accuracy       | ~96%  |

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



## Challenges Encountered and Solutions

### 1. Label Mapping Mismatch

#### Problem
The training dataset was organized into folders named:

```text
0, 1, 2, 3, ..., 42
```

Initially, the class folders were loaded using:

```python
self.classes = sorted(os.listdir(root_dir))
```

This caused the folders to be sorted alphabetically instead of numerically:

```text
['0', '1', '10', '11', '12', ..., '2', '20', ...]
```

As a result, the labels assigned during training did not match the official GTSRB class IDs used in the test dataset.

For example:

```text
Folder "16" -> Training Label 8
Folder "33" -> Training Label 27
Folder "38" -> Training Label 32
```

This led to incorrect evaluation results and a misleading confusion matrix.

#### Solution

The folder names were sorted numerically:

```python
self.classes = sorted(
    os.listdir(root_dir),
    key=lambda x: int(x)
)
```

This ensured that the training labels matched the official GTSRB class IDs.

---

### 2. Incorrect Test Dataset Implementation

#### Problem

The initial test dataset loader only returned images:

```python
return image
```

The official labels stored in `Test.csv` were not loaded, making it impossible to properly evaluate the model.

#### Solution

A custom test dataset class was implemented to read image paths and labels directly from the CSV file:

```python
label = int(row["ClassId"])
```

The dataset now returns:

```python
return image, label
```

allowing accurate evaluation on the official test set.

---

### 3. Confusion Matrix Appeared Incorrect

#### Problem

The confusion matrix showed strong off-diagonal patterns despite achieving approximately 92% accuracy.

This suggested that the model was performing poorly, which contradicted the reported accuracy.

#### Solution

After investigation, it was discovered that the model predictions used the internal training label encoding while the test labels used the official GTSRB class IDs.

The predictions were mapped back to the correct class IDs before generating the confusion matrix.

After fixing the label mapping, the confusion matrix displayed a strong diagonal pattern, confirming that the model was classifying traffic signs correctly.

---

### 4. Validation and Test Results Were Inconsistent

#### Problem

Validation accuracy was high, but sample predictions on the official test set appeared incorrect.

This made it difficult to determine whether the model or the evaluation pipeline was at fault.

#### Solution

Several debugging techniques were used:

- Printed class mappings used during training
- Compared sample predictions with CSV labels
- Verified image ordering in the test dataset
- Inspected the contents of `Test.csv`
- Compared validation and test label encodings

These checks helped identify the label encoding mismatch and ensured that training and testing used consistent class IDs.

---

## Key Lessons Learned

This project provided valuable hands-on experience with:

- Building custom PyTorch datasets
- Implementing image classification pipelines
- Data augmentation techniques
- CNN model development
- Debugging dataset and label mapping issues
- Working with real-world benchmark datasets
- Evaluating model performance using confusion matrices
- Troubleshooting inconsistencies between validation and test results

One of the most important lessons learned was that a model can achieve strong accuracy while still producing misleading evaluation results if label encodings are inconsistent across different parts of the machine learning pipeline. Careful verification of data loading and label mapping is essential for reliable model evaluation.

