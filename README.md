# **DrivingSim_DeepLearningModel**  
### **Pedestrian Threat Detection in Driving Simulations Using CNNs**  

## **Overview**  
This deep learning model classifies pedestrian proximity risk based on images captured from a driving simulator. Using pixel-wise labels for objects in the field of view, the model assesses pedestrian risk levels by analyzing the area occupied by pedestrians in each frame.  

## **Objective**  
The primary goal of this project is to develop a neural network capable of identifying and classifying pedestrian risk levels (low, medium, high) based on their proximity to the simulated vehicle. This can be useful for autonomous driving research, traffic safety assessments, and driver assistance system development.  

## **Dataset**  
- The dataset is sourced from the **DAVID dataset** in the **CARLA driving simulator**.
- Densely Annotated Video Driving (DAVID) Data Set
https://dataserv.ub.tum.de/index.php/s/m1596437?path=%2F
The DAVID data set consists of 28 video sequences of urban driving recorded in the CARLA simulator.
- Each frame includes pixel-wise labeled objects, enabling precise pedestrian segmentation.  
- The dataset is preprocessed to extract pedestrian bounding areas and classify risk levels accordingly.  

## **Methodology**  
**Preprocessing**: Extract pedestrian-related pixel information and normalize image data.  
# Pedestrian Risk Classifier

## Model Architecture

The `PedestrianRiskClassifier` is a PyTorch Lightning implementation designed to classify pedestrian risk levels in images.

### Core Components

#### 1. Object Detection Model
- Pre-trained Faster R-CNN with ResNet-50 backbone and Feature Pyramid Network (FPN)
- Configured to detect people (class 1 in COCO dataset)
- Used to extract pedestrian information from images

#### 2. Feature Extraction Backbone
- Supports multiple backbone options:
  - ResNet-50
  - ResNet-101
  - EfficientNet-B0
- Pre-trained on ImageNet (optional)
- Final classification layer replaced with an Identity layer to use as a feature extractor

#### 3. Classification Components
- Density-aware classifier that combines visual features with pedestrian density information

### Feature Processing

The model processes images in two parallel streams:

#### Pedestrian Detection Stream
- Extracts three key pedestrian features:
  - Normalized pedestrian count (number of pedestrians ÷ 10)
  - Total pedestrian area ratio (total area of pedestrians ÷ image area)
  - Average pedestrian size ratio (average pedestrian area ÷ image area)
- Only considers high-confidence detections (score > 0.7)

#### Visual Feature Stream
- Processes the whole image through the backbone network
- Extracts high-level visual features

These two streams are combined and fed into the density features classifier to produce the final risk classification.
![Risk Classification Confusion Matrix](/assets/visualization_results/v002_0044/activation_heatmap.png)
![Risk Classification Confusion Matrix](/assets/visualization_results/v002_0044/pedestrian_detection.png)
### Classification Output
- Three risk classes: Low, Medium, High
- Uses cross-entropy loss for training

### Training and Evaluation
- Implemented using PyTorch Lightning for structured training
- Includes accuracy metrics for training, validation, and testing
- Provides detailed evaluation metrics (precision, recall, F1-score) for each risk class
- Generates confusion matrices for performance visualization
- Uses Adam optimizer with learning rate scheduler (ReduceLROnPlateau)

## **Dependencies**  
**Dependencies include:**  
- Python 3.8  
- PyTorch  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib
- torch
- torchvision
- pytorch-lightning
- torchmetrics
- scikit-learn

## **Results**  

### Model Performance

Our pedestrian risk classifier achieved excellent performance metrics on the test dataset:

- **Accuracy**: 91.0%
- **Macro Average F1 Score**: 0.91

Detailed performance by risk category:

| Risk Level | Precision | Recall | F1 Score | Support |
|------------|-----------|--------|----------|---------|
| Low (0)    | 0.91      | 0.93   | 0.92     | 351     |
| Medium (1) | 0.86      | 0.89   | 0.87     | 356     |
| High (2)   | 0.96      | 0.91   | 0.93     | 370     |

### Error Analysis

The classifier misclassified only 96 examples out of 1077 (8.91% error rate). The error distribution shows the following patterns:

- **High → Medium**: 31 samples (32.3% of errors)
- **Medium → Low**: 28 samples (29.2% of errors)
- **Low → Medium**: 21 samples (21.9% of errors)
- **Medium → High**: 11 samples (11.5% of errors)
- **High → Low**: 3 samples (3.1% of errors)
- **Low → High**: 2 samples (2.1% of errors)

This distribution indicates that most errors occur between adjacent risk categories, with very few severe misclassifications (high-to-low or low-to-high).

### Confidence Analysis

The model showed relatively high confidence even in its misclassifications, with an average confidence score of 0.7188 on incorrect predictions. This suggests potential areas for calibration improvement in future iterations.

### Visualization

![Risk Classification Confusion Matrix](/assets/confusion_matrix.png)
![Risk Classification Confusion Matrix](/assets/multiclass_ROC_curve.png)
![Risk Classification Confusion Matrix](/assets/sample_pred.png)

### Key Insights

- The model demonstrates particularly strong performance in identifying high-risk scenarios (precision of 0.96)
- Adjacent category misclassifications comprise over 80% of errors, suggesting the model rarely makes severe misjudgments
- The balanced performance across categories (similar F1 scores) indicates robust classification regardless of risk level 

## **Contributors**  
- Nick Hansen
- Christopher Kuhn, Markus Hofbauer, Murong Xu, and Eckehard Steinbach
- https://ieeexplore.ieee.org/document/9506552

## **License**  
This project is licensed under the Apache 2.0 (LICENSE). 
maybe change this***
