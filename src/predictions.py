import torch
from network import SimpleSiamese

from loader import balance_dataset, control_pairs, subject_patient_pairs, shifted_subject_patient_pairs
import os
from visualizations import *
import argparse
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import numpy as np
from main import predict, generate_roc_curve

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese Network Operations")

    parser.add_argument('--model', type=str, choices=['SLO', 'MLO'],
                             help='Type of model architecture to use (custom or VGG16-based).', 
                             required=True)
    parser.add_argument('--model_path', type=str, help='Path to the model to load', required=True)
    parser.add_argument("--preop_dir", type=str, default='./data/processed/preop/BTC-preop', help=
                        "Path to the directory containing the preprocessed data")
    parser.add_argument("--mode", type=str, choices= ['sanity_check', 'augmented'] ,default='sanity_check', help="Mode of testing, sanity_check or augmented")
    parser.add_argument("--tumor_dir", type=str, default='./data/raw/preop/BTC-preop/derivatives/tumor_masks', help=
                        "Path to the directory containing suject dirs with tumor masks, relative is possible from project dir \
                        should contain sub-pat01, sub-pat02 etc. with tumor.nii in them")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    if args.mode == 'sanity_check':
        subject_images = control_pairs(proc_preop=args.preop_dir,
                    image_ids=['t1_ants_aligned.nii.gz'], skip=16, tumor_sensitivity=0.18)
    elif args.mode == 'augmented':
        subject_images = shifted_subject_patient_pairs(proc_preop=args.preop_dir, raw_tumor_dir=args.tumor_dir,
                    image_ids=['t1_ants_aligned.nii.gz'], skip=2, tumor_sensitivity=0.18)
        subject_images = balance_dataset(subject_images)

    print(f"Total number of images: {len(subject_images)}")
    ## No balancing, it is intentionally left unbalanced
    print("Number of similar batches:", len([x for x in subject_images if x['label'] == 1]))
    print("Number of dissimilar batches:", len([x for x in subject_images if x['label'] == 0]))
    
    if args.model == 'SLO':
        model_type = SimpleSiamese()
    elif args.model == 'MLO':
        pass
    
    ## using validation split to avoid overfitting
    test_loader = DataLoader(subject_images, batch_size=16, shuffle=False)
    model_params =  args.model_path.split("/")[-3]
    save_dir = f'./results/{model_params}/{args.mode}'
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    model_type.load_state_dict(torch.load(args.model_path))
    
    distances, labels = predict(model_type, test_loader, save_dir, device)

    thresholds = generate_roc_curve(distances, labels, save_dir)