# ENEN-645-Group-4 - Assignment 2
# Team members
- Taranvir Hundal
- Maciek Popik
- Mohammad Defaee
- Freya Rezaei

This project implements a multimodal garbage classification system that combines cellphone images with short natural-language descriptions to automatically determine the correct
waste stream. The system will classify items into green (compost), blue (recycling), black (garbage), or other (special disposal). 
In this final version, the model uses ResNet50 for image feature extraction and DistilBERT for processing textual data. A comparsion was made between ResNET50 and MobileNetV2, a lightweight CNN optimized for reduced computational cost. 
While MobileNetV2 may be sufficient for this task, ResNet50 offers greater representational capacity and acheived a higher classification accuracy at the expense of increased computation. 

# General Workflow 
1. Obtain the Dataset:
Download the datasets from the provided remote server. Here, we are using Onedrive remote server. The dataset size is approximately 15GB.

2. Run the Pipeline:
Execute the consolidated script group6_assignment2_final.ipynb to train and evaluate the model. This script:
- Loads training, validation, and test data.
- Trains the model, saving the best weights as final_resnet_adamW_model.pth.
- Evaluates the model on the test set, reporting overall accuracy, per-class accuracy, and displaying a confusion matrix.

# Directory Structure
# Installation and Dependencies 
Ensure you have the following prerequisites installed:

Python 3.x
PyTorch
Transformers
torchvision
scikit-learn
seaborn
matplotlib
Install the required packages using:
```
pip install torch torchvision transformers scikit-learn seaborn matplotlib
```
#Dataset Structure 
Place the following in the project root:
- CVPR_2026_dataset_Train
- CVPR_2026_dataset_Val
- CVPR_2026_dataset_Test

#Usage 
1. Update Dataset Paths: Modify the dataset paths in group4_assignment2_final.ipynb to point to your local or remote dataset location.
2. Run the Script: Execute the following command to start the training and evaluation process: python group4_assignment2_final.ipynb The script automatically detects and utilizes GPU, MPS (for Apple Silicon), or CPU based on availability.

#Model Details 
- Image Features: ResNet50
- Text Features: DistilBERT 
- Fusion Layers: Custom fully connected layers fuse a 512-dimensional image representation with a 512-dimensional text representation for final classification.
- Training: Uses the AdamW optimizer (learning rate = , weight decay = ) over  epochs.
- Evaluation: Computes overall accuracy, per-class accuracy, and displays a confusion matrix.

#Running on TALC
1. Transfer Files: Copy group6_assignment2_final.py (Not ipynb) to TALC.
2. Update Dataset Path: In group6_assignment2_final.py, update the dataset path to: /work/TALC/enel645_2026w/garbage_data
3. Create SLURM Job Files: Prepare a .slurm file to run group6_assignment2_final.py.
4. Execute: Run the job to train the model and generate results with final_resnet_adamW_model.pth.

#Results
After training and evaluation on the test set, the model achieved an overall accuracy of approximately %. Detailed per-class performance is as follows:

- Black: %
- Blue: %
- Green: %
- Other:

#Comparision with other Models 
1. Single vs Multi-model garbage system (Image or text model only)
2. ResNet50 vs MobileNetV2
  

