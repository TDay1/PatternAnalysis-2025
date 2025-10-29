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

## Improved U-Net

#### Background
In 2015, Ronneberger et al. [TODO CITE] published "U-Net: Convolutional Networks for Biomedical Image Segmentation", which introduced the novel "U-Net" convolution neural network architecture. This architecture incorporated the idea of adding "skip connections" to the encoder-decoder architecture, enabling more precise segmentations.

<img src="./images/original-unet-architecture.png" width="600">

*Figure 3: The original U-Net architecture. Diagram from the original paper [TODO CITE]*

The original U-Net's encoder consisted 5 levels, each using two 3x3 convolutions with an ReLU activation function. The decoder consisted of the same architecture, except it was mirrored, increasing the spatial feature using up-convolutions (transposed convolutions) and it also accepted the input from the skip connections at each level. 

#### Improved U-Net Architecture

In 2018, Isensee et al. release an "Improved U-Net" paper [TODO: CITE], that build upon the work on Renneberger at al. to create a U-Net-like network that could operate on three-dimensional data and achieve improved state of the art accuracy. While the Improved U-Net paper was originally intended for three-dimensional data, as per the task description (project 3) this report will adapt it for two-dimensional data. Therefore, the following description of the architecture will be for a two-dimensional variation of the Improved U-Net paper, rather than the original three-dimensional architecture.

<img src="./images/improved-unet-architecture.png" width="600">

*Figure 4: The Improved U-Net diagram, taken from the paper [TODO cite]*

**Encoder**
The encoder section improved upon the original U-Net implementation by introducing context modules, which are based on pre-activation residual blocks introduced by by He et al. in "Identity Mappings in Deep Residual Networks" [TODO CITE:  https://arxiv.org/pdf/1603.05027]. The Context modules consist of two layers containing instance normalisation, LeakyReLU, and a convolution, with a dropout layer in-between. The residual is taken at the start of the block and summed to the output of the two layers.


<img src="./images/context-module.svg" width="600">

*Figure 5: Diagram of the context module from the Improve U-Net paper [TODO CITE]*

With the exception of the final layer, the output of each context module is saved to later be passed to the decoder through a skip connection before finally being downscaled using a 3x3 convolution with a stride of 2.


**Decoder**
The decoder section also improves on that of the original U-Net paper. First, localisation modules are introduced. These modules take the previous layer's features, concatenated with the features received from the skip connection, as input. Then, it feeds this input through a 3x3 convolution followed by a 1x1 convolution that halfs the number of features.

The next component in the decoder is the upsampling module. This module aims to reduce the checkerboard artifacts often introduced by upscaling with transposed convolutions, It achieves this by simply upscaling the input by repeating each feature twice, before appying a stanfard convolution and activation function. 

The final contribution in the decoder is the addition of "deep supervision" to the U-Net architecture. Broadly speaking, this mechanism works by passing the output of the top three levels in the decoder through segmentation layers, upscaling them to the same resolution, and then summing them, giving the final segmented output.


**Loss function**
The original U-Net paper used a weighted cross-entropy loss function. They rationale behind this loss function was that the weights could be used to weight underrepresented classes higher, thereby balancing the loss from the different classes.

Improved U-Net replaces this loss function with multiclass Dice loss. This loss function was chosen to address the high degree of class imbalance often observed in medical datasets. Because this loss function computes the overlap on a per-class basis before then averaging, all classes are equally weighted, thereby solving this issue.

<img src="./images/loss-fn.png" width="200">

*Figure 7: The multiclass DICE loss used in Improved U-Net*
