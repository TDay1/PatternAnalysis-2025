# Project 3

**Author**: Thomas Day (46986791)

**Chosen Project**: Project 3 (Normal Difficulty)

---
## Introduction
The aim of this project is to adapt the "Improved UNet" architecture [CITE] to segment 2D Magnetic Resonance Imaging (MRI) scans of human hips into six distinct classes. This report will explore the model architecture, training process, and results of this task. Success will be measured by all labels having a minimum Dice-Sørensen (Dice) coeffecient of 0.75.

## Table of Contents

0. [Introduction](#Introduction)
1. [Table of Contents](#Table of Contents)
2. [Dataset](#Dataset)
3. Project Goal
4. Architecture
5. Data augmentation
6. Training process
7. Training Results
8. Discussion
9. Running instructions
10. References


## Dataset
As specified by the task sheet, the Improved UNet model will be trained on the HipMRI Study on prostate cancer [CITE] dataset. This dataset contains 12,660  greyscale images from MRI scans of the male human pelvis. These images were divided into train, validation, and test of sizes 11460, 660, and 540 respectively. 

| Split    | Number of samples | Purpose                                                                          |
|----------|:-----------------:|----------------------------------------------------------------------------------|
| Train    | 11460             | Used to fit the model.                                                           |
| Validate | 660               | Used to tune hyperparameters, monitor training progress, and detect overfitting. |
| Test     | 540               | Used to test how well the final model performs. Not used until after training.   |

*Figure 1: A table describing the train, validate, and test spits.*

In addition to the 12,660 MRI scan images are corresponding ground-truth segmentation labels, which assign any given pixel in any given image to one of six classes including background. Figure 2 describes these 6 classes below.

| Class ID | Label      | Example Ground Truth Segmentation    |
| -------- | ---------- | ------------------------------------ |
| 0        | Background | <img src="./images/readme-ground-truth-class-0.png" width="200"> |
| 1        | Body       | <img src="./images/readme-ground-truth-class-1.png" width="200"> |
| 2        | Bones      | <img src="./images/readme-ground-truth-class-2.png" width="200"> |
| 3        | Bladder    | <img src="./images/readme-ground-truth-class-3.png" width="200"> |
| 4        | Rectum     | <img src="./images/readme-ground-truth-class-4.png" width="200"> |
| 5        | Prostate   | <img src="./images/readme-ground-truth-class-5.png" width="200"> |

*Figure 2: A table demonstrating the classes from the HipMRI dataset [CITE]. Classes 0, 1, 2, 3, and 5 example taken from train image 567, class 4 example taken from train image 582*

