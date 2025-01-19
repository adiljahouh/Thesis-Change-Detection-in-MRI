import torch as torch
import torch.optim as optim
from network import SimpleSiamese, complexSiameseExt, DeepLabExtended
from loss_functions import contrastiveLoss, contrastiveThresholdLoss, \
    eval_feature_map, contrastiveThresholdMaskLoss, resize_tumor_to_label_dim
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
from numpy import ndarray
from torchvision.transforms import Compose
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
def predict(siamese_net: torch.nn.Module, test_loader: DataLoader, base_dir, device=torch.device('cuda'), model_type='SLO'):
    siamese_net.to(device)
    siamese_net.eval()  # Set the model to evaluation mode
    distances_list = []
    labels_list = []
    print("Doing predictions...")
    with torch.no_grad():
        for index, batch in enumerate(test_loader): 
            batch: dict[str, torch.Tensor]
            pre_batch: torch.Tensor = batch['pre'].float().to(device)
            post_batch: torch.Tensor = batch['post'].float().to(device)
            
            labels = batch['label'].to(device)
            assert pre_batch.shape == post_batch.shape, "Pre and post batch shapes do not match"
            
            first_conv: torch.Tensor
            second_conv: torch.Tensor
            third_conv: torch.Tensor
            # these are all batches
            first_conv, second_conv, third_conv = siamese_net(pre_batch, post_batch, mode='test')
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
                pre_image: ndarray = np.squeeze(pre_batch[batch_index].data.cpu().numpy())
                post_image: ndarray = np.squeeze(post_batch[batch_index].data.cpu().numpy())
                baseline = batch['baseline'][batch_index]
                label = labels[batch_index].item()  # Get the label for the i-th pair

                dist = (distance_1[batch_index], distance_2[batch_index], distance_3[batch_index])
                # print(f"Pair has distances of: {dist[0].item()}, {dist[1].item()}, {dist[2].item()}, label: {label}")
                # tumor maps should only be calculated for dissimilar pairs
                    
                distances_list.append(dist)
                labels_list.append(label)
                                # Save the heatmap
                if label == 0:  
                    filename = (
                        f"slice_{batch['pat_id'][batch_index]}_"
                        f"{'axial_' if batch['index_post'][0][batch_index] != -1 else ''}"
                        f"{batch['index_post'][0][batch_index] if batch['index_post'][0][batch_index] != -1 else ''}"
                        f"{'coronal_' if batch['index_post'][1][batch_index] != -1 else ''}"
                        f"{batch['index_post'][1][batch_index] if batch['index_post'][1][batch_index] != -1 else ''}"
                        f"{'sagittal_' if batch['index_post'][2][batch_index] != -1 else ''}"
                        f"{batch['index_post'][2][batch_index] if batch['index_post'][2][batch_index] != -1 else ''}_{label}.jpg"
                    )
                    save_dir = os.path.join(os.getcwd(), f'{base_dir}/heatmaps')
                    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
                    save_path = f'{save_dir}/{filename}'
                    pre_tumor = np.squeeze(batch['pre_tumor'][batch_index].data.cpu().numpy())  

                     
                    distance_map_2d_conv1 = return_upsampled_norm_distance_map(
                    first_conv[0][batch_index], first_conv[1][batch_index], dist_flag='l2', mode='bilinear')
                    distance_map_2d_conv2 = return_upsampled_norm_distance_map(
                    second_conv[0][batch_index], second_conv[1][batch_index], dist_flag='l2', mode='bilinear')
                    distance_map_2d_conv3 = return_upsampled_norm_distance_map(
                    third_conv[0][batch_index], third_conv[1][batch_index], dist_flag='l2', mode='bilinear')
                    
                    conv1_sharpened_pre = multiplicative_sharpening_and_filter(distance_map_2d_conv1, base_image=pre_image)
                    conv2_sharpened_pre = multiplicative_sharpening_and_filter(distance_map_2d_conv2, base_image=pre_image)
                    conv3_sharpened_pre = multiplicative_sharpening_and_filter(distance_map_2d_conv3, base_image=pre_image)
                    
                    conv1_sharpened_post = multiplicative_sharpening_and_filter(distance_map_2d_conv1, base_image=post_image)
                    conv2_sharpened_post = multiplicative_sharpening_and_filter(distance_map_2d_conv2, base_image=post_image)
                    conv3_sharpened_post = multiplicative_sharpening_and_filter(distance_map_2d_conv3, base_image=post_image)

                    visualize_multiple_fmaps_and_tumor_baselines(
                                    (np.rot90(pre_image), "Preoperative"), 
                                    (np.rot90(post_image), "Postoperative"), 
                                    (np.rot90(conv1_sharpened_pre), "First Layer Pre"), 
                                    (np.rot90(conv1_sharpened_post), "First layer post"), 
                                    (np.rot90(conv2_sharpened_pre), "Second layer pre"),
                                    (np.rot90(conv2_sharpened_post), "Second layer post"),
                                    (np.rot90(conv3_sharpened_pre), "Third layer pre"),
                                    (np.rot90(conv3_sharpened_post), "Third layer post"),
                                    (np.rot90(distance_map_2d_conv1), "Conv 1 Raw"), 
                                    (np.rot90(distance_map_2d_conv2), "Conv 2 Raw"),
                                    (np.rot90(distance_map_2d_conv3), "Conv 3 Raw"),
                                    (np.rot90(np.squeeze(baseline)), "Baseline method"), output_path=save_path, 
                                    tumor=np.rot90(pre_tumor), pre_non_transform=np.rot90(pre_image))
    return distances_list, labels_list

def train(siamese_net: torch.nn.Module, optimizer: Optimizer, criterion: torch.nn.Module,
          train_loader: DataLoader, val_loader: DataLoader, epochs=100, patience=3, 
          save_dir='./results/unassigned', device=torch.device('cuda')):
    
    siamese_net.to(device)
    print(f"Number of batches in training set: {len(train_loader)}")
    print(f"Number of batches in validation set: {len(val_loader)}")
    
    print("\nStarting training...")
    best_loss = float('inf')
    best_f1_score = float('-inf')
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
            pre_tumor_batch = batch['pre_tumor'].to(device)
            post_tumor_batch = batch['post_tumor'].to(device)
            
            # visualize_multiple_images((pre_batch[0].data.cpu().numpy().squeeze(), "Preoperative"), 
            # (pre_tumor_batch[0].data.cpu().numpy().squeeze(), "Preoperative Tumor"),
            # (post_batch[0].data.cpu().numpy().squeeze(), "Postoperative"),
            # (post_tumor_batch[0].data.cpu().numpy().squeeze(), "Postoperative Tumor"),
            # output_path=f"{os.getcwd()}/src/checking_tumor.png")
            # return
            assert pre_batch.shape == post_batch.shape, "Pre and post batch shapes do not match"
            siamese_net.train()  # switch to training mode

            optimizer.zero_grad()
            first_conv, second_conv, third_conv = siamese_net(pre_batch, post_batch)
            # TODO: tumor shift check and check control pair handling
            ## TODO: MERGE TUMORS? Or only pass post tumor to the loss function?
            # Resize tumor to match the dimensions of the convolutional layers
            tumor_resized_to_first_conv = resize_tumor_to_label_dim(
                pre_tumor_batch, first_conv[0].data.cpu().numpy().shape[2:])
            tumor_resized_to_second_conv = resize_tumor_to_label_dim(
                pre_tumor_batch, second_conv[0].data.cpu().numpy().shape[2:])
            tumor_resized_to_third_conv = resize_tumor_to_label_dim(
                pre_tumor_batch, third_conv[0].data.cpu().numpy().shape[2:])
            ## TODO: need conv distance for each conv layer and then visualize it
            
            ## tumors used for loss function but USE only POST? not both -> focus on change
            ## THen visualize it before passing it to the loss function
            loss_1 = criterion(first_conv[0], first_conv[1], tumor_resized_to_first_conv)
            loss_2 = criterion(second_conv[0], second_conv[1], tumor_resized_to_second_conv)
            loss_3 = criterion(third_conv[0], third_conv[1], tumor_resized_to_third_conv)
            loss: torch.Tensor = loss_1 + loss_2 + loss_3

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()            
        # Validation loop
        siamese_net.eval()  # switch to evaluation mode
        epoch_f1_scores = 0.0
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                batch_f1_scores = 0.0
                batch: dict[str, torch.Tensor]
                pre_batch: torch.Tensor = batch['pre'].float().to(device)
                post_batch: torch.Tensor = batch['post'].float().to(device)
                pre_tumor_batch: torch.Tensor = batch['pre_tumor'].float().to(device)

                assert pre_batch.shape == post_batch.shape, "Pre and post batch shapes do not match"
                first_conv, second_conv, third_conv = siamese_net(pre_batch, post_batch)
                for batch_index in range(pre_batch.size(0)):                            
                    # distance_map_1 = return_upsampled_distance_map(first_conv[0][batch_index], first_conv[1][batch_index],
                    #                                             dist_flag='l2', mode='bilinear')
                    distance_map_2 = return_upsampled_distance_map(second_conv[0][batch_index], second_conv[1][batch_index],
                                                                dist_flag='l2', mode='bilinear')
                    # distance_map_3 = return_upsampled_distance_map(third_conv[0][batch_index], third_conv[1][batch_index],
                    #                                             dist_flag='l2', mode='bilinear')
                    f1_score, validation = eval_feature_map(pre_tumor_batch.cpu().numpy()[batch_index][0], distance_map_2.data.cpu().numpy()[0][0], 0.30, 
                                                            beta=0.8)
                    batch_f1_scores += f1_score
                batch_f1_scores /= pre_batch.size(0) 
                epoch_f1_scores += batch_f1_scores        
        # Calculate average loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_f1_score = epoch_f1_scores / len(val_loader)
        #print(f"Average sample loss for epoch {epoch+1}: Train Loss: {epoch_train_loss/total_train_samples}, Val Loss: {epoch_val_loss/total_val_samples}")
        print(f'Epoch [{epoch+1}/{epochs}], Average Train Loss: {avg_train_loss:.4f}, Average f1 score: {avg_f1_score:.4f}')
        
        # Check for improvement in validation loss
        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
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
    interp = torch.nn.Upsample(size=(256, 256), mode='bilinear')

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
        criterion = contrastiveLoss(margin=args.margin, dist_flag=args.dist_flag)
        transform=Compose([
                    T.ToTensor()])
    elif args.loss == 'TCL':
        criterion = contrastiveThresholdMaskLoss(hingethresh=args.threshold, margin=args.margin)
        transform = Compose([
                    T.ToTensor(),
                    ShiftImage(max_shift_x=50, max_shift_y=50),
                    # T.RandomVerticalFlip(),
                    # T.RandomHorizontalFlip(),
                    # RotateImage(angle=random.randint(0, 180), padding_mode='border', align_corners=True)
                    ]
        )
            
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
    # balance subject_images based on label
    
    print(f"Total number of images: {len(subject_images)}")
    subject_images: list[dict] = balance_dataset(subject_images)
    print(f"Total number of total pairs after balancing: {len(subject_images)}")
    train_subject_images, val_subject_images, test_subject_images = random_split(subject_images, (0.6, 0.2, 0.2))
    
    optimizer = optim.Adam(model_type.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model_type.parameters(), lr=0.01, momentum=0.9)

    ## collates the values into one tensor per key
    train_loader = DataLoader(train_subject_images, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_subject_images, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_subject_images, batch_size=args.batch_size, shuffle=False)

    model_params =  f'{args.model}_{args.dist_flag}_'\
            f'lr-{args.lr}_marg-{args.margin}_thresh-{args.threshold}_loss-{args.loss}'
    save_dir = f'./results/{model_params}/train_test'
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    _ = train(model_type, optimizer, criterion, train_loader=train_loader, val_loader=val_loader, 
            epochs=args.epochs, patience=args.patience, 
            save_dir=save_dir, device=device)

    
    distances, labels = predict(model_type, test_loader, base_dir =save_dir, device=device, model_type=args.model)

    # take the conv distance distance from each tuple
    thresholds = generate_roc_curve([d[0].item() for d in distances], labels, save_dir, "_conv1")
    thresholds = generate_roc_curve([d[1].item() for d in distances], labels, save_dir, "_conv2")
    thresholds = generate_roc_curve([d[2].item() for d in distances], labels, save_dir, "_conv3")