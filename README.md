# ENEN-645-Group-4 - Assignment 2
# Multimodal Garbage Classification System
# Team members
- Taranvir Hundal
- Maciek Popik
- Mohammad Defaee
- Freya Rezaei

This project implements a multimodal garbage classification system that combines cellphone images with short natural-language descriptions (derived from filenames) to automatically determine the correct waste stream.
The system classifies items into four categories:
* Green – Compost
* Blue – Recycling
* Black – Garbage
* TTR / Other – Special disposal

In the final version, the model uses:
* ResNet50 for image feature extraction
* DistilBERT (uncased) for textual feature extraction
* A custom fusion MLP for multimodal classification

A comparison was performed between ResNet50 and MobileNetV2 (a lightweight CNN optimized for reduced computational cost). While MobileNetV2 achieved similar accuracy in some runs, ResNet50 provided:
* Slightly higher peak accuracy
* More consistent results (lower variance between runs)
* Stronger separation of visually similar classes (e.g., Black vs Blue)

MobileNetV2 did not provide a significant speed advantage in this setup, therefore ResNet50 was selected for the final model.

# General Workflow 
1. Obtain the Dataset:
Download the datasets from the provided remote server.
* Dataset size: ~15GB
* Hosted on OneDrive for Google Colaboratory access
* Also processed locally on lab hardware with sufficient GPU memory

The dataset is already divided by folders into:
* Training
* Validation
* Test

2. Run the Pipeline:
Execute the final notebook: `ENEN645_Assignment_2_Group_4.ipynb`

This notebook:
* Loads training, validation, and test datasets
* Performs two-phase transfer learning:
    - Phase 1: Train fusion head only
    - Phase 2: Fine-tune pretrained encoders
* Applies:
    - Class weighting for imbalance
    - ReduceLROnPlateau scheduler
    - Early stopping
* Saves best checkpoints:
    - mm_head_only.pth
    - mm_finetuned.pth
* Evaluates the final model on the test set
* Reports:
    - Overall accuracy
    - Per-class metrics (Sensitivity, Specificity, Precision, F1)
    - Confusion matrix
    - Qualitative correct and incorrect examples

# Installation and Dependencies 
Ensure you have the following prerequisites installed:

torch
torchvision
transformers
numpy
matplotlib

Install the required packages using:
```
pip install torch torchvision transformers numpy matplotlib
```

# Dataset Structure 
Place the following in the project root:
- CVPR_2024_dataset_Train
- CVPR_2024_dataset_Val
- CVPR_2024_dataset_Test

# Model Architecture
Image Encoder
* ResNet50 (ImageNet pretrained)
* Final FC layer removed
* 2048-dimensional feature vector projected to 512-dimensional embedding

Text Encoder
* DistilBERT (distilbert-base-uncased)
* Mean-pooled token embeddings
* 768-dimensional representation

Fusion Strategy
* Image (512-dim) + Text (768-dim)
* Concatenated into a 1280-dimensional vector
* Passed through a custom MLP classifier
  
# Training Strategy
Two-phase transfer learning:

Phase 1 – Fusion Head Training
* ResNet50 + DistilBERT frozen
* Only fusion MLP trained
* Optimizer: AdamW
* Higher learning rate

Phase 2 – Fine-Tuning
* All layers unfrozen
* Differential learning rates:
    - DistilBERT: very small LR
    - ResNet50: small LR
    - Fusion head: slightly higher LR
* ReduceLROnPlateau scheduler
* Early stopping on validation loss

Class imbalance handled via weighted CrossEntropyLoss.


# Evaluation
The model reports:
* Overall test accuracy
* Per-class:
    - Sensitivity (Recall)
    - Specificity
    - Precision
    - F1 score
* Confusion matrix
* Balanced qualitative samples of:
    - Correct classifications
    - Misclassifications

Particular attention is given to confusion between Black and Blue classes, which represent visually overlapping waste categories.