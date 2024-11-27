import torch
import torch.optim as optim
from network import SimpleSiamese, complexSiameseExt
from loss_functions import ConstractiveLoss, ConstractiveThresholdHingeLoss
from loader import aertsDataset, remindDataset, balance_dataset
from transformations import ShiftImage, RotateImage
import os
from visualizations import *
import argparse
from torch.utils.data import random_split, DataLoader, ConcatDataset
import torch.nn.functional as F
import numpy as np
from torch.optim.optimizer import Optimizer
from torchvision import transforms as T
from torchvision.transforms import Compose,InterpolationMode 

## segmentated data https://openneuro.org/datasets/ds001226/versions/5.0.0

## warp ants on raw/ses-preop skull data
## this turns it into processed/ants

## ants script ants/registrationSynQuick used rigid and affine
## ants apply transform

## check fsl and if we brought post to pre we can just load tumor and mannually check
## ELSE
## if it doesnt work just use t1.mif use mrconvert to save as nii.gz


### Post t1_ants is aligned to preop t1_ants
### fact checked all patients, 
### its not all aligned to the tumor correctly but my alignment was worse

## aligned them but skull is different which can cause issues
## write in docu we registered tumors using antsApplyTransforms


## using non-aligned since theyre almost fully aligned bythemselves, did align postop to preop though
## Aligned and normalized the tumor using nilearn transformations such that python handles it properly
## One voxel input is too little information so im looking into patches, tried voxel based but didnt make sense
## padded them

## using images and tumor mask but ratio is not 1:1
## filtering low info slices
## balancing classes
## skipped CV can reintroduce it later

## https://medium.com/data-science-in-your-pocket/understanding-siamese-network-with-example-and-codes-e7518fe02612
def predict(siamese_net: nn.Module, test_loader: DataLoader, base_dir, device=torch.device('cuda'), model_type='SLO'):
    siamese_net.to(device)
    siamese_net.eval()  # Set the model to evaluation mode
    distances_list = []
    labels_list = []
    with torch.no_grad():
        for index, batch in enumerate(test_loader): 
            batch: dict[str, torch.Tensor]
            pre_batch: torch.Tensor = batch['pre'].float().to(device)
            post_batch: torch.Tensor = batch['post'].float().to(device)
            assert pre_batch.shape == post_batch.shape, "Pre and post batch shapes do not match"
            # Add channel dimension (greyscale image)
            ## if len(pre_batch.size()) == 3:
            # pre_batch = pre_batch.unsqueeze(1)
            # post_batch = post_batch.unsqueeze(1)
            

            labels = batch['label'].to(device)
            if model_type == 'SLO':
                output1, output2 = siamese_net(pre_batch, post_batch)
                output1: torch.Tensor
                output2: torch.Tensor
                flattened_batch_t0 = output1.view(output1.size(0), -1)  
                flattened_batch_t1 = output2.view(output2.size(0), -1)
                assert flattened_batch_t0.size(0) == flattened_batch_t1.size(0), "Flattened batch sizes do not match"
                distance = F.pairwise_distance(flattened_batch_t0, flattened_batch_t1, p=2)

            elif model_type == 'MLO':
                first_conv: torch.Tensor
                second_conv: torch.Tensor
                third_conv: torch.Tensor
                first_conv, second_conv, third_conv = siamese_net(pre_batch, post_batch)

                flattened_batch_conv1_t0 = first_conv[0].view(first_conv[0].size(0), -1)
                flattened_batch_conv1_t1 = first_conv[1].view(first_conv[1].size(0), -1)
                distance_1 = F.pairwise_distance(flattened_batch_conv1_t0, flattened_batch_conv1_t1, p=2)

                flattened_batch_conv2_t0 = second_conv[0].view(second_conv[0].size(0), -1)
                flattened_batch_conv2_t1 = second_conv[1].view(second_conv[1].size(0), -1)
                distance_2 = F.pairwise_distance(flattened_batch_conv2_t0, flattened_batch_conv2_t1, p=2)

                flattened_batch_conv3_t0 = third_conv[0].view(third_conv[0].size(0), -1)
                flattened_batch_conv3_t1 = third_conv[1].view(third_conv[1].size(0), -1)
                distance_3 = F.pairwise_distance(flattened_batch_conv3_t0, flattened_batch_conv3_t1, p=2)
                assert distance_1.size(0) == distance_2.size(0) == distance_3.size(0), "Distance sizes do not match"
            
            # Iterate over the batch
            for batch_index in range(pre_batch.size(0)):
                baseline = batch['baseline'][batch_index]
                label = labels[batch_index].item()  # Get the label for the i-th pair
                if model_type == 'MLO':
                    dist = (distance_1[batch_index], distance_2[batch_index], distance_3[batch_index])
                    
                    _, distance_map_2d_conv1 = single_layer_similar_heatmap_visual(
                    first_conv[0][batch_index], first_conv[1][batch_index], dist_flag='l2', mode='bilinear')
                    _, distance_map_2d_conv2 = single_layer_similar_heatmap_visual(
                    second_conv[0][batch_index], second_conv[1][batch_index], dist_flag='l2', mode='bilinear')
                    _, distance_map_2d_conv3 = single_layer_similar_heatmap_visual(
                    third_conv[0][batch_index], third_conv[1][batch_index], dist_flag='l2', mode='bilinear')
                    print(f"Pair has distances of: {dist[0].item()}, {dist[1].item()}, {dist[2].item()}, label: {label}")
                elif model_type == 'SLO':
                    dist = distance[batch_index]
                    _, distance_map_2d = single_layer_similar_heatmap_visual(output1[batch_index], 
                    output2[batch_index], dist_flag='l2', mode='bilinear')
                    print(f"Pair has a distance of: {dist.item()}, label: {label}")


                distances_list.append(dist)
                labels_list.append(label)
                filename = (
                    f"slice_{batch['pat_id'][batch_index]}_"
                    f"{'axial_' if batch['index_post'][0][batch_index] != -1 else ''}"
                    f"{batch['index_post'][0][batch_index] if batch['index_post'][0][batch_index] != -1 else ''}"
                    f"{'coronal_' if batch['index_post'][1][batch_index] != -1 else ''}"
                    f"{batch['index_post'][1][batch_index] if batch['index_post'][1][batch_index] != -1 else ''}"
                    f"{'sagittal_' if batch['index_post'][2][batch_index] != -1 else ''}"
                    f"{batch['index_post'][2][batch_index] if batch['index_post'][2][batch_index] != -1 else ''}_{label}.jpg"
                )

                # Save the heatmap
                save_dir = os.path.join(os.getcwd(), f'{base_dir}/heatmaps')
                os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
                save_path = f'{save_dir}/{filename}'
                pre_image = np.rot90(np.squeeze(batch['pre'][batch_index]))
                post_image = np.rot90(np.squeeze(batch['post'][batch_index]))
                if label == 0:
                    pre_non_transform = np.rot90(np.load(batch["pre_path"][batch_index])['data'])
                    tumor = np.rot90(np.load(batch["tumor_path"][batch_index])['data'])
                else:
                    pre_non_transform = None
                    tumor = None

                if model_type == 'SLO':
                    merge_images((pre_image, "pre"), (post_image, "post"), (np.rot90(distance_map_2d), "First Layer"),
                                 np.rot90(np.squeeze(baseline)), output_path=save_path,
                                 tumor=tumor, pre_non_transform=pre_non_transform)
                elif model_type == 'MLO':
                    merge_images((pre_image, "Preoperative"), (post_image, "Postoperative"), (np.rot90(distance_map_2d_conv1), "First Layer"), 
                                 (np.rot90(distance_map_2d_conv2), "Second Layer"), (np.rot90(distance_map_2d_conv3), "Third Layer"),
                                  (np.rot90(np.squeeze(baseline)), "Baseline method"), output_path=save_path, 
                                  tumor=tumor, pre_non_transform=pre_non_transform)
    return distances_list, labels_list

def train(siamese_net: nn.Module, optimizer: Optimizer, criterion: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs=100, patience=3, 
          save_dir='./results/unassigned', device=torch.device('cuda')):
    
    siamese_net.to(device)
    print(f"Number of batches in training set: {len(train_loader)}")
    print(f"Number of batches in validation set: {len(val_loader)}")
    
    print("\nStarting training...")
    best_loss = float('inf')
    consecutive_no_improvement = 0
    
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        total_train_samples = 0
        for index, batch in enumerate(train_loader):
            ## each batch is a dict with pre, post, label etc. and collated (merged) values from
            ## each value in the batch
            batch: dict[str, torch.Tensor]
            
            assert batch['pre'].shape == batch['post'].shape, "Pre and post batch shapes do not match"
            assert type(batch['pre']) == type(batch['post']) == torch.Tensor, "Pre or post is not a tensor, use transform ToTensor()  in the dataSet or unsqueeze(0) after loading each batch"
            
            pre_batch = batch['pre'].float().to(device)
            post_batch = batch['post'].float().to(device)

            ## add channel dimension because its just collated(merged) 2D numpy arrays
            
            # pre_batch = pre_batch.unsqueeze(1)
            # post_batch = post_batch.unsqueeze(1)

            assert pre_batch.shape == post_batch.shape, "Pre and post batch shapes do not match"
            siamese_net.train()  # switch to training mode

            optimizer.zero_grad()
            label_batch = batch['label'].to(device)

            if args.model == 'SLO':
                output1, output2 = siamese_net(pre_batch, post_batch)
                loss: torch.Tensor = criterion(output1, output2, label_batch)
            elif args.model == 'MLO':

                first_conv, second_conv, third_conv = siamese_net(pre_batch, post_batch)
                loss_1 = criterion(first_conv[0], first_conv[1], label_batch)
                loss_2 = criterion(second_conv[0], second_conv[1], label_batch)
                loss_3 = criterion(third_conv[0], third_conv[1], label_batch)
                loss: torch.Tensor = loss_1 + loss_2 + loss_3

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            total_train_samples += pre_batch.size(0)
            
        # Validation loop
        siamese_net.eval()  # switch to evaluation mode
        epoch_val_loss = 0.0
        total_val_samples = 0
        with torch.no_grad():
            for index, batch in enumerate(val_loader):

                batch: dict[str, torch.Tensor]
                pre_batch: torch.Tensor = batch['pre'].float().to(device)
                post_batch: torch.Tensor = batch['post'].float().to(device)

                # pre_batch = pre_batch.unsqueeze(1)
                # post_batch = post_batch.unsqueeze(1)
                assert pre_batch.shape == post_batch.shape, "Pre and post batch shapes do not match"
                label_batch = batch['label'].to(device)
                if args.model == 'SLO':
                    output1, output2 = siamese_net(pre_batch, post_batch)
                    loss: torch.Tensor = criterion(output1, output2, label_batch)
                elif args.model == 'MLO':
                    first_conv, second_conv, third_conv = siamese_net(pre_batch, post_batch)
                    loss_1 = criterion(first_conv[0], first_conv[1], label_batch)
                    loss_2 = criterion(second_conv[0], second_conv[1], label_batch)
                    loss_3 = criterion(third_conv[0], third_conv[1], label_batch)
                    loss: torch.Tensor = loss_1 + loss_2 + loss_3
                epoch_val_loss += loss.item()
                total_val_samples += pre_batch.size(0)
        
        # Calculate average loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        #print(f"Average sample loss for epoch {epoch+1}: Train Loss: {epoch_train_loss/total_train_samples}, Val Loss: {epoch_val_loss/total_val_samples}")
        print(f'Epoch [{epoch+1}/{epochs}], Average Train Loss: {avg_train_loss:.4f}, Average Val Loss: {avg_val_loss:.4f}')
        
        # Check for improvement in validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            consecutive_no_improvement = 0
            # Save the best model
            save_path = os.path.join(save_dir, "model.pth")
            torch.save(siamese_net.state_dict(), save_path)
            print(f'Saved best model to {save_path}')
        else:
            consecutive_no_improvement += 1
            if consecutive_no_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1} as no improvement for {patience} consecutive epochs.')
                break
    return best_loss           

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese Network Operations")
    parser.add_argument('--model', type=str, choices=['SLO', 'MLO'],
                             help='Type of model architecture to use (custom or VGG16-based).', 
                             required=True)
    parser.add_argument("--aerts_dir", type=str, default='./data/processed/preop/BTC-preop', help=
                        "Path to the directory containing the preprocessed subject dirs FROM AERTS, relative is possible from project dir\
                            should contain sub-pat01, sub-pat02 etc. with a specific nifti image id anywhere down the tree")
    parser.add_argument("--remind_dir", type=str, default='./data/ReMIND_dataset/ReMIND-Manifest-Sept-2023/ReMIND', help=
                        "Path to the directory containing the preprocessed subject dirs FROM REMIND, relative is possible from project dir\
                            should contain remind-001, remind-002 etc. with a specific nifti image_id anywhere down the tree")
    parser.add_argument("--tumor_dir", type=str, default='./data/raw/preop/BTC-preop/derivatives/tumor_masks', help=
                        "Path to the directory containing suject dirs with tumor masks, relative is possible from project dir \
                        should contain sub-pat01, sub-pat02 etc. with tumor.nii in them")
    parser.add_argument("--slice_dir", type=str, default='./data/2D/', help="location for slices to be saved and loaded")
    parser.add_argument("--loss", type=str, choices=['CL', 'TCL'], default="TCL", help=
                        "Type of loss function to use (constractive or thresholded constractive)")
    parser.add_argument("--dist_flag", type=str, choices=['l2', 'l1', 'cos'], default='l2', help=
                        "Distance flag to use for the loss function (l2, l1, or cos)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--patience", type=int, default=8, help="Patience for early stopping")
    parser.add_argument("--margin", type=float, default=5.0, help="Margin for dissimilar pairs")
    parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for similar pairs, prevents overfit")
    parser.add_argument("--skip", type=int, default=1, help=" Every xth slice to take from the image, if 1 take all. Saves memory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--load_slices", action="store_true", help="Load slices instead of full 3D image")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    if args.loss == 'CL':
        criterion = ConstractiveLoss(margin=args.margin, dist_flag=args.dist_flag)
        transform=Compose([
                    T.ToTensor()])
    elif args.loss == 'TCL':
        criterion = ConstractiveThresholdHingeLoss(hingethresh=args.threshold, margin=args.margin)
        transform = Compose([
                    T.ToTensor(),
                    ShiftImage(max_shift_x=50, max_shift_y=50),
                    T.RandomVerticalFlip(),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(60, InterpolationMode=InterpolationMode.BILINEAR, fill=None, expand=True)])
            
    if args.model == 'SLO':
        ## Always call T.ToTensor()
        aertsImages = aertsDataset(proc_preop=args.aerts_dir, 
                  raw_tumor_dir=args.tumor_dir,
                  image_ids=['t1_ants_aligned.nii.gz'], skip=args.skip, tumor_sensitivity=0.30,
                  save_dir='./data/2D/',
                  transform=transform, load_slices=args.load_slices) 
        print("Aerts dataset loaded")
        remindImages = remindDataset(preop_dir=args.remind_dir, 
                    image_ids=['t1_aligned_stripped'], save_dir=args.slice_dir,
                    skip=args.skip, tumor_sensitivity=0.30, transform=transform, load_slices=args.load_slices)
        subject_images = ConcatDataset([aertsImages, remindImages])       
        model_type = SimpleSiamese()
    elif args.model == 'MLO':
        aertsImages = aertsDataset(proc_preop=args.aerts_dir, 
                  raw_tumor_dir=args.tumor_dir, save_dir=args.slice_dir,
                  image_ids=['t1_ants_aligned.nii.gz'], skip=args.skip, 
                  tumor_sensitivity=0.30,transform=transform, load_slices=args.load_slices)
        print("Aerts dataset loaded")
        remindImages = remindDataset(preop_dir=args.remind_dir, 
                    image_ids=['t1_aligned_stripped'], save_dir=args.slice_dir,
                    skip=args.skip, tumor_sensitivity=0.30, transform=transform, load_slices=args.load_slices)
        subject_images = ConcatDataset([aertsImages, remindImages])
        model_type = complexSiameseExt()
    pass
    # balance subject_images based on label
    
    print(f"Total number of images: {len(subject_images)}")
    subject_images: list[dict] = balance_dataset(subject_images)
    print(f"Total number of total pairs after balancing: {len(subject_images)}")
    train_subject_images, val_subject_images, test_subject_images = random_split(subject_images, (0.6, 0.2, 0.2))
    
    optimizer = optim.Adam(model_type.parameters(), lr=args.lr)

    ## collates the values into one tensor per key
    train_loader = DataLoader(train_subject_images, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_subject_images, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_subject_images, batch_size=args.batch_size, shuffle=False)

    model_params =  f'{args.model}_{args.dist_flag}_'\
            f'lr-{args.lr}_marg-{args.margin}_thresh-{args.threshold}_loss-{args.loss}'
    save_dir = f'./results/{model_params}/train_test'
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    best_loss = train(model_type, optimizer, criterion, train_loader=train_loader, val_loader=val_loader, 
            epochs=args.epochs, patience=args.patience, 
            save_dir=save_dir, device=device)

    
    distances, labels = predict(model_type, test_loader, base_dir =save_dir, device=device, model_type=args.model)

    if args.model == 'SLO':
        thresholds = generate_roc_curve(distances, labels, save_dir)
    elif args.model == 'MLO':
        # take the first distance from each tuple
        thresholds = generate_roc_curve([d[0].item() for d in distances], labels, save_dir, "_conv1")
        thresholds = generate_roc_curve([d[1].item() for d in distances], labels, save_dir, "_conv2")
        thresholds = generate_roc_curve([d[2].item() for d in distances], labels, save_dir, "_conv3")