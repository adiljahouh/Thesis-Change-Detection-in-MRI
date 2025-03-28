# Project Guide

Welcome to my **Change Detection in MRI** repository! This guide will help you understand how to use this project and its model script effectively.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Setup Instructions](#setup-instructions)
3. [Using the Model Script](#using-the-model-script)
4. [Project Workflow](#project-workflow)
5. [Flowchart](#flowchart)

---

## Introduction

This project is designed to detect image-based changes in intraoperative tumor resection. Inspired by Guo et al. (2018).

---

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/Thesis-Change-Detection-in-MRI.git
    cd Thesis-Change-Detection-in-MRI
    ```

2. Set up the environment:

    **Option A — Using Conda (recommended):**
    ```bash
    conda env create -f environment.yml
    conda activate change-detection-mri
    ```

    **Option B — Using pip:**
    ```bash
    pip install -r requirements.txt
    ```

3. Configure the environment:
    - Make sure you have CUDA setup
    - Make sure you have all drivers installed
    - Check by calling ```nvidia-smi``` in CMD LINE
    - Install the correct PyTorch https://pytorch.org/get-started/locally/

---
## Data Setup
1. Unzip ```2D.zip``` in the ```./data``` directory.


2.  Optionally: unzip ```ReMIND_dataset.zip``` in the ```./data``` directory as well, if you want to use the slicing function and generate your own slices (with your own tumor thresholds)

3. Now we should have:

```
myProject/
├── data/
│   ├── 2D/
│       ├── overview/
│       ├── ReMIND-102_slice_sagittal_99_change_map_label_0.npz
│       ├── ReMIND-102_slice_sagittal_99_post_label_0.npz
│       ├── ReMIND-102_slice_sagittal_99_post_tumor_label_0.npz
│       ├── ReMIND-102_slice_sagittal_99_pre_label_0.npz
│       ├── ReMIND-102_slice_sagittal_99_pre_tumor_label_0.npz
│       ├── ...
│   ├── ReMIND_dataset/
│       ├── ReMIND-Manifest-Sept-2023/
│           ├── ReMIND
│               ├── ReMIND-002
│               ├── ReMIND-004
│               ├── ...
│   └── test/
├── environment.yml
├── modelscript.sh
├── notebooks/
├── output.log
├── predict.sh
├── README.MD
├── requirements.txt
├── results/
├── scripts/
├── src/
```
## Training the model

1. To train, run the model script:
    ```bash
    bash model_script.sh
    ```

2. It is highly recommended to use --load_slices in your train and prediction script.

3. Configuring skip, patience, load_slices parameter and thresholds in the modelscript file can enhance training.


---
## Testing a pretrained model
1. To test the model, run:
    ```bash 
    bash predict.sh
    ```
2. This grabs the pretrained ```model.pth``` in ```results/MLO_l2_lr-0.001_marg-4.0_thresh-0.7_loss-TCL/train_test/model.pth```

3. It then executes ```predictions.py``` on the 2D sliced data, applying augmentations and such. 


---
## Project Workflow

The project follows this workflow:
1. Optional: Preprocess the data (2D generation)
2. Train the model using the provided script.
3. Evaluate the model's performance.
4. Generate predictions.

---

## Flowchart

Below is the flowchart representing the project workflow during inference

![Project Workflow](notebooks/images/flowchart_inference.png)

---

Feel free to reach out if you have any questions or issues!