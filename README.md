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
1. **Preprocessing**: Extract pedestrian-related pixel information and normalize image data.  
2. **Model Architecture**:  
   - A Convolutional Neural Network (CNN) processes frames to detect pedestrians.  
   - The model classifies pedestrian risk based on pixel area thresholds.  
3. **Training & Evaluation**:  
   - The model is trained on labeled simulation frames.  
   - Performance is evaluated using accuracy, precision, recall, and F1-score.  

## **Dependencies**  
**Dependencies include:**  
- Python 3.x  
- PyTorch  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  

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
