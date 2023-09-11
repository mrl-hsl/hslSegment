# Semantic Segmentation of Field Lines and Grass for Soccer Robot

## Overview

This project performs semantic segmentation on images from a soccer field to identify field lines and grass. The goal is to enable a soccer robot to better understand its environment and localize itself on the field. 

Semantic segmentation classifies every pixel in an image, allowing the robot to precisely locate field elements like lines, goals, corner flags, etc. This provides more detailed environmental information compared to object detection or classification alone.

## Approach

The segmentation model is based on a convolutional neural network architecture optimized for mobile devices. Key aspects:

- Lightweight depthwise separable convolutions for efficiency
- Encoder-decoder structure to capture context and localize objects 
- Multi-scale processing with dilated convolutions to capture objects at different sizes

The model is trained on a dataset of soccer field images labeled with pixel-level segmentation masks. Data augmentation (flips, crops) improves model robustness. 

Loss function weights pixels based on class frequency to handle imbalance between dominant grass and smaller field line classes.

## Data

- 600 1280x720 JPEG images of soccer fields 
- Corresponding segmentation masks in JPEG format
- Classes:
  - Grass (RGB 0,255,0)
  - Field lines (RGB 255,0,0)
  - Background (RGB 0,0,0)
- 80/20 train/test split
- Converted to TFRecord format for efficient input pipeline

## Training

- Adam optimizer
- Mixed precision training for faster computation
- Trained for 15 epochs at batch size 8
- Learning rate decayed from 1e-3 to 1e-5 over training

## Evaluation

- Mean IoU: 0.92
  - Grass IoU: 0.95
  - Lines IoU: 0.81
- Runs at 22 FPS on Nvidia Jetson TX2

The model successfully segments field elements accurately, providing the robot detailed understanding of its environment for navigation and play.

## Repository Contents

- `network.py`: Model architecture 
- `dataProvider.py`: Data loading and preprocessing
- `master.py`: Main training script
- `converterToTFRecords.py`: Script to convert dataset to TFRecords
- `logs/`: TensorBoard logs
- `dataSet/`: Image and mask samples
