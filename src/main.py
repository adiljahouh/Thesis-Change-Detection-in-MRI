import torch
import torch.optim as optim
from network import SaimeseTwoDim
from loss_functions import ConstractiveLoss
from loader import imagePairs, balance_dataset
import os
from visualizations import merge_images, generate_roc_curve, single_layer_similar_heatmap_visual
import argparse
import cv2
from torch.utils.data import random_split, DataLoader, Subset
import torch.nn.functional as F
import numpy as np
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

## work in patches, but that would mean a lot of labels. Maybe aggregate them to have some sort of probability?
## or can we just go back to images? 

## using images and tumor mask but ratio is not 1:1
## filtering low info slices
## balancing classes
## skipped CV can reintroduce it later

## https://medium.com/data-science-in-your-pocket/understanding-siamese-network-with-example-and-codes-e7518fe02612
def predict(siamese_net, test_loader, threshold=0.3):
    siamese_net.to(device)
    siamese_net.eval()  # Set the model to evaluation mode
    distances_list = []
    labels_list = []
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            input1 = batch['pre'].float().to(device)
            input2 = batch['post'].float().to(device)
            
            # Add channel dimension (greyscale image)
            input1 = input1.unsqueeze(1)
            input2 = input2.unsqueeze(1)

            labels = batch['label'].to(device)
            output1, output2 = siamese_net(input1, input2)
            flattened_batch_t0 = output1.view(output1.size(0), -1)  
            flattened_batch_t1 = output2.view(output2.size(0), -1)

            distance = F.pairwise_distance(flattened_batch_t0, flattened_batch_t1, p=2)
            for i in range(distance.size(0)):
                label = labels[i].item()  # Get the label for the i-th pair

                dist = distance[i].item()  # Get the distance for the i-th pair
                distances_list.append(dist)
                labels_list.append(label)
                prediction = dist < threshold  # Determine if the pair is similar based on the threshold
                if prediction:
                    print(f"Pair {i} is similar with a distance of: {dist}, label: {label}")
                else:
                    print(f"Pair {i} is dissimilar with a distance of: {dist}, label: {label}")

            # Visualize the similarity heatmap
                filename = (
                    f"slice_{batch['pat_id'][i]}_"
                    f"{'axial' if batch['index_post'][0][i] != -1 else ''}_"
                    f"{batch['index_post'][0][i].item() if batch['index_post'][0][i] != -1 else ''}"
                    f"{'coronal' if batch['index_post'][1][i] != -1 else ''}_"
                    f"{batch['index_post'][1][i].item() if batch['index_post'][1][i] != -1 else ''}"
                    f"{'sagittal' if batch['index_post'][2][i] != -1 else ''}_"
                    f"{batch['index_post'][2][i].item() if batch['index_post'][2][i] != -1 else ''}.jpg"
                )
                heatmap, out1_trans, out2_trans = single_layer_similar_heatmap_visual(output1[i], output2[i], 'l2')
                # Save the heatmap
                save_dir = os.path.join(os.getcwd(), f'./data/heatmaps/twodim')
                os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
                save_path = f'{save_dir}/{filename}'
                pre_image = np.rot90(np.squeeze(batch['pre'][i]))
                post_image = np.rot90(np.squeeze(batch['post'][i]))
                merge_images(pre_image, post_image, np.rot90(heatmap), save_path)
    return distances_list, labels_list

def train(siamese_net, optimizer, criterion, train_loader, val_loader, epochs=100, patience=3, 
          save_dir='models', model_name='masked.pth', device=torch.device('cuda')):
    siamese_net.to(device)
    print(f"Number of samples in training set: {len(train_loader)}")
    print(f"Number of samples in validation set: {len(val_loader)}")
    
    print("\nStarting training...")
    best_loss = float('inf')
    consecutive_no_improvement = 0
    
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        for index, batch in enumerate(train_loader):
            pre_batch = batch['pre'].float().to(device)
            post_batch = batch['post'].float().to(device)
            # Add channel dimension (greyscale image)
            pre_batch = pre_batch.unsqueeze(1)
            post_batch = post_batch.unsqueeze(1)

            siamese_net.train()  # switch to training mode
            label = batch['label'].to(device)
            output1, output2 = siamese_net(pre_batch, post_batch)
            loss = criterion(output1, output2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            
        # Validation loop
        siamese_net.eval()  # switch to evaluation mode
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                pre_batch = batch['pre'].float().to(device)
                post_batch = batch['post'].float().to(device)

                pre_batch = pre_batch.unsqueeze(1)
                post_batch = post_batch.unsqueeze(1)

                output1, output2 = siamese_net(pre_batch, post_batch)
                label = batch['label'].to(device)
                loss = criterion(output1, output2, label)
                epoch_val_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Check for improvement in validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            consecutive_no_improvement = 0
            # Save the best model
            save_path = os.path.join(save_dir, model_name)
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

    parser.add_argument('--model', type=str, choices=['custom', 'vgg16'],
                             help='Type of model architecture to use (custom or VGG16-based).', 
                             required=True)
    parser.add_argument("--dist_flag", type=str, choices=['l2', 'l1', 'cos'], required=True, help=
                        "Distance flag to use for the loss function (l2, l1, or cos)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin of the constractive loss")
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    save_dir = f'./models/{args.model}'
    ##TODO: add root path to args.pars because this wont run on server
    subject_images = imagePairs(proc_preop='./data/processed/preop/BTC-preop', 
                  raw_tumor_dir='./data/raw/preop/BTC-preop/derivatives/tumor_masks',
                  image_ids=['t1_ants_aligned.nii.gz'], skip=2, tumor_sensitivity=0.15)
    # balance subject_images based on label
    print(f"Total number of images: {len(subject_images)}")
    print("Number of similar pairs:", len([x for x in subject_images if x['label'] == 1]))
    print("Number of dissimilar pairs:", len([x for x in subject_images if x['label'] == 0]))

    subject_images = balance_dataset(subject_images)
    print(f"Total number of images after balancing: {len(subject_images)}")
    train_subject_images, val_subject_images, test_subject_images = random_split(subject_images, (0.6, 0.2, 0.2))

    print("Number of similar pairs:", len([x for x in subject_images if x['label'] == 1]))
    print("Number of dissimilar pairs:", len([x for x in subject_images if x['label'] == 0]))
    
    if args.model == 'custom':
        model_type = SaimeseTwoDim()
    # elif args.model == 'vgg16':
    #     model_type = SiameseVGG3D()
    criterion = ConstractiveLoss(margin=args.margin, dist_flag=args.dist_flag)
    optimizer = optim.Adam(model_type.parameters(), lr=args.lr)

    ## using validation split to avoid overfitting
    train_loader = DataLoader(train_subject_images, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_subject_images, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_subject_images, batch_size=16, shuffle=False)
    #TODO: Add betters samples by tuning tumor sensitivity
    #TODO: validate images BY PLOTTING THEM NEXT TO HEATMAP
    #TODO: FIX RIM ISSUE IN HEATMAPS
    #TODO: MASKS HAVE INTENSITY AT POINTS THEY SHOULDNT!
    #TODO: WE CANT JUST LABEL 0 or 1 in PAT based on tumor, there could be changes in the brain that arent tumor related

    #TODO: mtrix calc for evaluation


    best_loss = train(model_type, optimizer, criterion, train_loader=train_loader, val_loader=val_loader, 
        epochs=args.epochs, patience=args.patience, 
        save_dir=save_dir, model_name=f'{args.model}_{args.dist_flag}_'\
        f'lr-{args.lr}_marg-{args.margin}.pth', device=device)
    
    distances, labels = predict(model_type, test_loader, args.margin)

    thresholds = generate_roc_curve(distances, labels, f"./models/{args.model}")