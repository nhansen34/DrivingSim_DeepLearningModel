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
- fill this in once complete***   

## **Contributors**  
- Nick Hansen
- Christopher Kuhn, Markus Hofbauer, Murong Xu, and Eckehard Steinbach
- https://ieeexplore.ieee.org/document/9506552

## **License**  
This project is licensed under the Apache 2.0 (LICENSE). 
maybe change this***
