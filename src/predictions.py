import torch
from network import SimpleSiamese, complexSiameseExt, DeepLabExtended, DeepLabV3

from loader import balance_dataset, control_pairs, aertsDataset, remindDataset
import os
from visualizations import *
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose
import torchvision.transforms as T
from torch.utils.data import ConcatDataset
from transformations import ShiftImage, RotateImage
from main import predict, generate_roc_curve

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese Network Operations")

    parser.add_argument('--model', type=str, choices=['SLO', 'MLO'],
                             help='Type of model architecture to use (custom or VGG16-based).', 
                             default='MLO')
    parser.add_argument('--model_path', type=str, help='Path to the model to load', required=True)
    parser.add_argument("--preop_dir", type=str, default='./data/processed/preop/BTC-preop', help=
                        "Path to the directory containing the preprocessed data")
    parser.add_argument("--mode", type=str, choices= ['sanity_check', 'augmented'] ,default='augmented', help="Mode of testing, sanity_check or augmented")
    parser.add_argument("--aerts_dir", type=str, default='./data/processed/preop/BTC-preop', help=
                        "Path to the directory containing the preprocessed subject dirs FROM AERTS, relative is possible from project dir\
                            should contain sub-pat01, sub-pat02 etc. with a specific nifti image id anywhere down the tree")
    parser.add_argument("--remind_dir", type=str, default='./data/ReMIND_dataset/ReMIND-Manifest-Sept-2023/ReMIND', help=
                        "Path to the directory containing the preprocessed subject dirs FROM REMIND, relative is possible from project dir\
                            should contain remind-001, remind-002 etc. with a specific nifti image_id anywhere down the tree")
    parser.add_argument("--tumor_dir", type=str, default='./data/raw/preop/BTC-preop/derivatives/tumor_masks', help=
                        "Path to the directory containing suject dirs with tumor masks, relative is possible from project dir \
                        should contain sub-pat01, sub-pat02 etc. with tumor.nii in them")
    parser.add_argument("--slice_dir", type=str, default='./data/test/', help="location for slices to be saved and loaded")
    parser.add_argument("--folder_name", type=str, default='inference', help="Name of the folder to save the results")
    args = parser.parse_args()
    model_details = args.model_path.split('/')[-3]
    dist_flag = model_details.split('_')[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    # if args.mode == 'sanity_check':
    #     subject_images = control_pairs(proc_preop=args.preop_dir,
    #                 image_ids=['t1_ants_aligned.nii.gz'], skip=16, tumor_sensitivity=0.18)
    # elif args.mode == 'augmented':
    if args.model == 'SLO':
        transform=Compose([
                    T.ToTensor()])
        aertsImages = aertsDataset(proc_preop=args.aerts_dir, 
                raw_tumor_dir=args.tumor_dir,
                image_ids=['t1_ants_aligned.nii.gz'], skip=50, tumor_sensitivity=0.30,
                save_dir='./data/2D/',
                transform=transform, load_slices=True) 
        print("Aerts dataset loaded")
        remindImages = remindDataset(preop_dir=args.remind_dir, 
                    image_ids=['t1_aligned_stripped'], save_dir=args.slice_dir,
                    skip=50, tumor_sensitivity=0.30, transform=transform, load_slices=True)
        subject_images = ConcatDataset([aertsImages, remindImages])       
        model_type = SimpleSiamese()
    elif args.model == 'MLO':
        transform = Compose([
            T.ToTensor(),
            ShiftImage(max_shift_x=50, max_shift_y=50),
            # T.RandomVerticalFlip(),
            # T.RandomHorizontalFlip(),
            # RotateImage(angle=random.randint(0, 180), padding_mode='border', align_corners=True)
            ]
        )
        # remindImages = aertsDataset(proc_preop=args.aerts_dir, 
        #         raw_tumor_dir=args.tumor_dir, save_dir=args.slice_dir,
        #         image_ids=['t1_ants_aligned.nii.gz'], skip=50, 
        #         tumor_sensitivity=0.30,transform=transform, load_slices=True)
        print("Aerts dataset loaded")
        remindImages = remindDataset(preop_dir=args.remind_dir, 
                    image_ids=['t1_aligned_stripped'], save_dir=args.slice_dir,
                    skip=1, tumor_sensitivity=0.30, transform=transform, load_slices=True)
        subject_images = remindImages
        model_type = DeepLabV3()

    print(f"Total number of images: {len(subject_images)}")
    ## No balancing, it is intentionally left unbalanced
    subject_images: list[dict] = balance_dataset(subject_images)
    
    print("Number of similar pairs:", len([x for x in subject_images if x['label'] == 1]))
    print("Number of dissimilar pairs:", len([x for x in subject_images if x['label'] == 0]))
 
    
    ## using validation split to avoid overfitting
    test_loader = DataLoader(subject_images, batch_size=16, shuffle=False)
    model_params =  args.model_path
    save_dir = os.path.join(os.path.dirname(args.model_path), args.folder_name)
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    model_type.load_state_dict(torch.load(args.model_path))
    
    distances, labels, f1_score = predict(model_type, test_loader, save_dir, device, dist_flag=dist_flag)
    if args.model == 'SLO':
        thresholds = generate_roc_curve(distances, labels, save_dir)
    elif args.model == 'MLO':
        # take the conv distance distance from each tuple
        # thresholds = generate_roc_curve([d[0].item() for d in distances], labels, save_dir, "_conv1")
        # thresholds = generate_roc_curve([d[1].item() for d in distances], labels, save_dir, "_conv2")
        # thresholds = generate_roc_curve([d[2].item() for d in distances], labels, save_dir, "_conv3")    # take the conv distance distance from each tuple
        thresholds = generate_roc_curve([d[0].item() for d in distances], labels, save_dir, f"_conv1_{f1_score}")
        thresholds = generate_roc_curve([d[1].item() for d in distances], labels, save_dir, f"_conv2_{f1_score}")
        thresholds = generate_roc_curve([d[2].item() for d in distances], labels, save_dir, f"_conv3_{f1_score}")