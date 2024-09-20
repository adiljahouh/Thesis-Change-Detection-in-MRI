import torch
import torch.optim as optim
from network import SimpleSiamese
from loss_functions import ConstractiveLoss, ConstractiveThresholdHingeLoss
from loader import balance_dataset, control_pairs
import os
from visualizations import *
import argparse
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import numpy as np
from main import predict, generate_roc_curve

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese Network Operations")

    parser.add_argument('--model', type=str, choices=['custom', 'vgg16'],
                             help='Type of model architecture to use (custom or VGG16-based).', 
                             required=True)
    parser.add_argument('--model_path', type=str, help='Path to the model to load', required=True)
    parser.add_argument("--preop_dir", type=str, default='./data/processed/preop/BTC-preop', help=
                        "Path to the directory containing the preprocessed data")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    subject_images = control_pairs(proc_preop=args.preop_dir,
                  image_ids=['t1_ants_aligned.nii.gz'], skip=4)
    print(f"Total number of images: {len(subject_images)}")

    subject_images = balance_dataset(subject_images)
    print(f"Total number of images after balancing: {len(subject_images)}")
    print("Number of similar batches:", len([x for x in subject_images if x['label'] == 1]))
    print("Number of dissimilar batches:", len([x for x in subject_images if x['label'] == 0]))
    
    if args.model == 'custom':
        model_type = SimpleSiamese()
    elif args.model == 'vgg16':
        pass
    
    ## using validation split to avoid overfitting
    test_loader = DataLoader(subject_images, batch_size=16, shuffle=False)
    model_params =  args.model_path.split("/")[-1]


    model_type.load_state_dict(torch.load(args.model_dir))
    
    distances, labels = predict(model_type, test_loader, f"{model_params}/sanity_check", args.margin)

    thresholds = generate_roc_curve(distances, labels, f"./results/{model_params}/sanity_check")