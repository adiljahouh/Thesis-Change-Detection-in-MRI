import torch as torch
import torch.optim as optim
from network import complexSiameseExt, DeepLabExtended
from loss_functions import contrastiveLoss, \
    eval_feature_map, contrastiveThresholdMaskLoss, resize_tumor_to_feature_map, compute_f_score, compute_iou
from loader import aertsDataset, remindDataset, balance_dataset
from distance_measures import threshold_by_zscore_std
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
def predict(siamese_net: torch.nn.Module, test_loader: DataLoader, 
            base_dir, device, dist_flag):
    siamese_net.to(device)
    siamese_net.eval()  # Set the model to evaluation mode
    distances_list = []
    labels_list = []
    f1_scores_list = []
    print("Doing predictions...")
    with torch.no_grad():
        f_score_conv3_total = 0.0
        f_score_baseline_total = 0.0
        f_score_baseline_z_total = 0.0
        
        mean_iou_conv3_total = 0.0
        mean_iou_baseline_total = 0.0
        mean_iou_baseline_z_total = 0.0
        
        conv3_precision_total = 0.0
        baseline_precision_total = 0.0
        baseline_z_precision_total = 0.0
        
        conv3_recall_total = 0.0
        baseline_recall_total = 0.0
        baseline_z_recall_total = 0.0

        shift_tensor = ShiftImage()
        rotate_tensor = RotateImage()
        for index, batch in enumerate(test_loader): 
            batch: dict[str, torch.Tensor]
            pre_batch: torch.Tensor = batch['pre'].float().to(device)
            post_batch: torch.Tensor = batch['post'].float().to(device)
            change_map_gt_batch: torch.Tensor = batch['change_map'].float().to(device)
            baseline_batch: torch.Tensor = batch['baseline'].float().to(device)
            
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
            batch_f1_scores = 0.0
            batch_baseline_f1_scores = 0.0
            batch_baseline_z_f1_scores = 0.0
            
            batch_miou_score_conv3 = 0.0
            batch_miou_score_baseline = 0.0
            batch_miou_score_baseline_z = 0.0
            
            batch_precision_conv3 = 0.0
            batch_recall_conv3 = 0.0
            
            batch_precision_baseline = 0.0
            batch_recall_baseline = 0.0
            
            batch_precision_baseline_z = 0.0
            batch_recall_baseline_z = 0.0
            disimilair_pairs = 0
            for batch_index in range(pre_batch.size(0)):
                pre_image: ndarray = np.squeeze(pre_batch[batch_index].data.cpu().numpy())
                post_image: ndarray = np.squeeze(post_batch[batch_index].data.cpu().numpy())
                baseline: ndarray = np.squeeze(baseline_batch[batch_index].data.cpu().numpy())
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
                    baseline_save_dir = os.path.join(os.getcwd(), f'{base_dir}/baselines')
                    save_dir = os.path.join(os.getcwd(), f'{base_dir}/fmaps')
                    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
                    os.makedirs(baseline_save_dir, exist_ok=True)
                    fmap_path = f'{save_dir}/{filename}'
                    baseline_path = f'{baseline_save_dir}/{filename}'
                    change_map_gt = np.squeeze(batch['change_map'][batch_index].data.cpu().numpy()) 
                    shift_values = (batch['shift_x'][batch_index], batch['shift_y'][batch_index])
                    rotation_angle = batch['rotation_angle'][batch_index]
                    pre_tumor = np.load(batch['tumor_path'][batch_index])['data']
                    
                    ## didnt want to keep all tumors in memory so is shift only test ones
                    post_tumor_unshifted_path = batch['residual_path'][batch_index]
                    post_tumor_unshifted = np.load(post_tumor_unshifted_path)['data']
                    post_tumor_unshifted_tensor = torch.tensor(post_tumor_unshifted, dtype=torch.float32)
                    post_tumor = post_tumor_unshifted_tensor
                    if shift_values != (0, 0):
                        post_tumor = shift_tensor(post_tumor, shift=shift_values)
                    if int(rotation_angle.item()) != 0:
                        post_tumor = rotate_tensor(post_tumor.unsqueeze(0), angle=rotation_angle)
                        post_tumor = np.squeeze(post_tumor.data.cpu().numpy())                    
                
                    distance_map_2d_conv3 = return_upsampled_norm_distance_map(
                    third_conv[0][batch_index], third_conv[1][batch_index], dist_flag=dist_flag, mode='bilinear')
                  
                    # baseline_significant = np.where(baseline > 0.30, 1, 0)
                    baseline = normalize_np_array(baseline)
                    baseline_masked = (baseline > 0.5).astype(np.uint8)
                    baseline_z_scored = threshold_by_zscore_std(baseline, threshold=4)
                    # baseline_99th_percentile = threshold_by_percentile(baseline, percentile=99)
                    f1_score_conv3, mean_miou_score_conv3, conv3_prec, conv3_recall = eval_feature_map(change_map_gt_batch.cpu().numpy()[batch_index][0], distance_map_2d_conv3, 0.30,
                                                            beta=1)
                    f1_scores_list.append(f1_score_conv3)
                    #f1_score_baseline, mean_miou_score_baseline, _, _ = eval_feature_map(change_map_gt_batch.cpu().numpy()[batch_index][0], baseline_z_scored, 0.30, beta=1)
                    f1_score_baseline, baseline_prec, baseline_recall = compute_f_score(tumor_seg=change_map_gt_batch.cpu().numpy()[batch_index][0], 
                                                        pred_mask=baseline_masked, beta=1)
                    mean_miou_score_baseline = compute_iou(tumor_seg=change_map_gt_batch.cpu().numpy()[batch_index][0],
                                                        pred_mask=baseline_masked)
                    
                    f1_score_baseline_z_scored, baseline_prec_z, baseline_recall_z = compute_f_score(tumor_seg=change_map_gt_batch.cpu().numpy()[batch_index][0],
                                                        pred_mask=baseline_z_scored, beta=1)
                    mean_miou_score_baseline_z_scored = compute_iou(tumor_seg=change_map_gt_batch.cpu().numpy()[batch_index][0],
                                                        pred_mask=baseline_z_scored)
                    
                    
                    conv3_sharpened_post = multiplicative_sharpening_and_filter(distance_map_2d_conv3, base_image=post_image, alpha=4)
                    batch_f1_scores += f1_score_conv3
                    batch_baseline_f1_scores += f1_score_baseline
                    batch_baseline_z_f1_scores += f1_score_baseline_z_scored
                    
                    batch_miou_score_conv3 += mean_miou_score_conv3
                    batch_miou_score_baseline += mean_miou_score_baseline
                    batch_miou_score_baseline_z += mean_miou_score_baseline_z_scored
                    
                    batch_precision_conv3 += conv3_prec
                    batch_recall_conv3 += conv3_recall
                    
                    batch_precision_baseline += baseline_prec
                    batch_recall_baseline += baseline_recall
                    
                    batch_precision_baseline_z += baseline_prec_z
                    batch_recall_baseline_z += baseline_recall_z
                    
                    disimilair_pairs += 1
                    try:
                        visualize_change_detection(
                            (np.rot90(baseline_masked), f"Fixed threshold prediction $\Delta \hat{{T}}_{{thresh}}$", f"F1={f1_score_baseline:.2f}, IoU={mean_miou_score_baseline:.2f}", None),
                            (np.rot90(baseline_z_scored), f"Z-scored prediction $\Delta \hat{{T}}_{{z-score}}$", f"F1={f1_score_baseline_z_scored:.2f}, IoU={mean_miou_score_baseline_z_scored:.2f}", None),
                            preoperative=(np.rot90(pre_image), np.rot90(pre_tumor)),  
                            postoperative=(np.rot90(post_image), np.rot90(post_tumor)),
                            ground_truth=(np.rot90(post_image), np.rot90(change_map_gt)),
                            output_path=baseline_path
                        )
                        visualize_change_detection(
                            (np.rot90(distance_map_2d_conv3), f"RiA prediction $\hat{{T}}_{{model}}$", f"F1={f1_score_conv3:.2f}, IoU={mean_miou_score_conv3:.2f}", np.rot90(conv3_sharpened_post)),
                            # (np.rot90(baseline_masked), f"Fixed threshold prediction $\Delta \hat{{T}}_{{thresh}}$", f"F1={f1_score_baseline:.2f}, IoU={mean_miou_score_baseline:.2f}", None),
                            # (np.rot90(baseline_z_scored), f"Z-scored prediction $\Delta \hat{{T}}_{{z-score}}$", f"F1={f1_score_baseline_z_scored:.2f}, IoU={mean_miou_score_baseline_z_scored:.2f}", None),
                            preoperative=(np.rot90(pre_image), np.rot90(pre_tumor)),  
                            postoperative=(np.rot90(post_image), np.rot90(post_tumor)),
                            ground_truth=(np.rot90(post_image), np.rot90(change_map_gt)),
                            output_path=fmap_path,
                            show_gt=True
                        )
                    except Exception as e:
                        print(f"Error in visualization: {e}")
                        print(f"Error type: {type(e).__name__}")
            batch_f1_scores /= disimilair_pairs
            batch_baseline_f1_scores /= disimilair_pairs
            batch_baseline_z_f1_scores /= disimilair_pairs
            
            batch_miou_score_conv3 /= disimilair_pairs
            batch_miou_score_baseline /= disimilair_pairs
            batch_miou_score_baseline_z /= disimilair_pairs
            
            batch_precision_conv3 /= disimilair_pairs
            batch_recall_conv3 /= disimilair_pairs
            
            batch_precision_baseline /= disimilair_pairs
            batch_recall_baseline /= disimilair_pairs
            
            batch_precision_baseline_z /= disimilair_pairs
            batch_recall_baseline_z /= disimilair_pairs
            # get batch average and add to total
            
            f_score_conv3_total += batch_f1_scores
            f_score_baseline_total += batch_baseline_f1_scores
            f_score_baseline_z_total += batch_baseline_z_f1_scores
            
            mean_iou_conv3_total += batch_miou_score_conv3
            mean_iou_baseline_total += batch_miou_score_baseline
            mean_iou_baseline_z_total += batch_miou_score_baseline_z
            
            conv3_precision_total += batch_precision_conv3
            conv3_recall_total += batch_recall_conv3
            
            baseline_precision_total += batch_precision_baseline
            baseline_recall_total += batch_recall_baseline
            
            baseline_z_precision_total += batch_precision_baseline_z
            baseline_z_recall_total += batch_recall_baseline_z
            
        f_score_conv3_total /= len(test_loader)
        f_score_baseline_total /= len(test_loader)
        f_score_baseline_z_total /= len(test_loader)
        
        mean_iou_conv3_total /= len(test_loader)
        mean_iou_baseline_total /= len(test_loader)
        mean_iou_baseline_z_total /= len(test_loader)
        
        conv3_precision_total /= len(test_loader)
        conv3_recall_total /= len(test_loader)
        
        baseline_precision_total /= len(test_loader)
        baseline_recall_total /= len(test_loader)
        
        baseline_z_precision_total /= len(test_loader)
        baseline_z_recall_total /= len(test_loader)
        
        with open(f'{base_dir}/results.txt', 'w') as f:
            f.write(f"Average f1 score for conv3: {f_score_conv3_total}\n")
            f.write(f"Average f1 score for baseline: {f_score_baseline_total}\n")
            f.write(f"Average f1 score for baseline z-scored: {f_score_baseline_z_total}\n")
            
            f.write(f"Average miou score for conv3: {mean_iou_conv3_total}\n")
            f.write(f"Average miou score for baseline: {mean_iou_baseline_total}\n")
            f.write(f"Average miou score for baseline z-scored: {mean_iou_baseline_z_total}\n")
            
            f.write(f"Average precision for conv3: {conv3_precision_total}\n")
            f.write(f"Average recall for conv3: {conv3_recall_total}\n")
            
            f.write(f"Average precision for baseline: {baseline_precision_total}\n")
            f.write(f"Average recall for baseline: {baseline_recall_total}\n")
            
            f.write(f"Average precision for baseline z-scored: {baseline_z_precision_total}\n")
            f.write(f"Average recall for baseline z-scored: {baseline_z_recall_total}\n")
            
        print(f"Average f1 score for conv3: {f_score_conv3_total:.2f}")
        print(f"Average f1 score for baseline: {f_score_baseline_total:.2f}")
        print(f"Average f1 score for baseline z-scored: {f_score_baseline_z_total:.2f}")
        
        print(f"Average miou score for conv3: {mean_iou_conv3_total:.2f}")
        print(f"Average miou score for baseline: {mean_iou_baseline_total:.2f}")
        print(f"Average miou score for baseline z-scored: {mean_iou_baseline_z_total:.2f}")
        
        print(f"Average precision for conv3: {conv3_precision_total:.2f}")
        print(f"Average recall for conv3: {conv3_recall_total:.2f}")    
        
        print(f"Average precision for baseline: {baseline_precision_total:.2f}")
        print(f"Average recall for baseline: {baseline_recall_total:.2f}")
        
        print(f"Average precision for baseline z-scored: {baseline_z_precision_total:.2f}")
        print(f"Average recall for baseline z-scored: {baseline_z_recall_total:.2f}")
        
    return distances_list, labels_list, round(f_score_conv3_total, 2), round(mean_iou_conv3_total, 2), f1_scores_list

def train(siamese_net: torch.nn.Module, optimizer: Optimizer, criterion: torch.nn.Module,
          train_loader: DataLoader, val_loader: DataLoader, epochs, patience, 
          save_dir, device, dist_flag):
    
    siamese_net.to(device)
    print(f"Number of batches in training set: {len(train_loader)}")
    print(f"Number of batches in validation set: {len(val_loader)}")
    
    best_loss = float('inf')
    best_f1_score = float('-inf')
    best_val_loss = float('inf')
    consecutive_no_improvement = 0
    
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        for index, train_batch in enumerate(train_loader):
            ## each batch is a dict with pre, post, label etc. and collated (merged) values from
            ## each value in the batch
            train_batch: dict[str, torch.Tensor]
            
            assert train_batch['pre'].shape == train_batch['post'].shape, "Pre and post train_batch shapes do not match"
            assert type(train_batch['pre']) == type(train_batch['post']) == torch.Tensor, "Pre or post is not a tensor, use transform ToTensor()  in the dataSet or unsqueeze(0) after loading each train_batch"
            
            pre_train_batch = train_batch['pre'].float().to(device)
            post_train_batch = train_batch['post'].float().to(device)
            # pre_tumor_train_batch = train_batch['pre_tumor'].to(device)
            post_tumor_train_batch = train_batch['change_map'].to(device)
            assert pre_train_batch.shape == post_train_batch.shape, "Pre and post train_batch shapes do not match"
            siamese_net.train()  # switch to training mode

            optimizer.zero_grad()
            first_conv_train, second_conv_train, third_conv_train = siamese_net(pre_train_batch, post_train_batch)
            # TODO: tumor shift check and check control pair handling
            ## TODO: MERGE TUMORS? Or only pass post tumor to the loss function?
            # Resize tumor to match the dimensions of the conv_trainolutional layers
            tumor_resized_to_first_conv_train = resize_tumor_to_feature_map(
                post_tumor_train_batch, first_conv_train[0].data.cpu().numpy().shape[2:])
            tumor_resized_to_second_conv_train = resize_tumor_to_feature_map(
                post_tumor_train_batch, second_conv_train[0].data.cpu().numpy().shape[2:])
            tumor_resized_to_third_conv_train = resize_tumor_to_feature_map(
                post_tumor_train_batch, third_conv_train[0].data.cpu().numpy().shape[2:])
            ## TODO: need conv_train distance for each conv_train layer and then visualize it
            
            ## tumors used for loss function but USE only POST? not both -> focus on change
            ## THen visualize it before passing it to the loss function

            loss_1 = criterion(first_conv_train[0], first_conv_train[1], tumor_resized_to_first_conv_train)
            loss_2 = criterion(second_conv_train[0], second_conv_train[1], tumor_resized_to_second_conv_train)
            loss_3 = criterion(third_conv_train[0], third_conv_train[1], tumor_resized_to_third_conv_train)
            # np.save(os.path.join(os.getcwd(), "pre_train.npy"), third_conv_train[0].data.cpu().numpy())
            # np.save(os.path.join(os.getcwd(), "post_train.npy"), third_conv_train[1].data.cpu().numpy())
            loss: torch.Tensor = loss_1 + loss_2 + loss_3
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()      
                  
        # Validation loop TODO: back to eval after debug
        siamese_net.eval()  # switch to evaluation mode
        epoch_f1_scores = 0.0
        with torch.no_grad():
            for index, val_batch in enumerate(val_loader):
                val_batch: dict[str, torch.Tensor]
                pre_val_batch: torch.Tensor = val_batch['pre'].float().to(device)
                post_val_batch: torch.Tensor = val_batch['post'].float().to(device)
                # pre_tumor_val_batch: torch.Tensor = val_batch['pre_tumor'].float().to(device)
                post_tumor_val_batch: torch.Tensor = val_batch['change_map'].float().to(device)
                assert pre_val_batch.shape == post_val_batch.shape, "Pre and post val_batch shapes do not match"
                first_conv_val, second_conv_val, third_conv_val = siamese_net(pre_val_batch, post_val_batch)
                # print("val stats")
                # print(first_conv_val[0].min(), first_conv_val[1].min(),second_conv_val[0].min(), second_conv_val[1].min(), third_conv_val[0].min(), third_conv_val[1].min())
                # print(first_conv_val[0].max(), first_conv_val[1].max(),second_conv_val[0].max(), second_conv_val[1].max(), third_conv_val[0].max(), third_conv_val[1].max())
                ## CHECK REGULAR LOSS
                ##################################################################################
                ###
                tumor_resized_to_first_conv_val = resize_tumor_to_feature_map(
                post_tumor_val_batch, first_conv_val[0].data.cpu().numpy().shape[2:])
                tumor_resized_to_second_conv_val = resize_tumor_to_feature_map(
                    post_tumor_val_batch, second_conv_val[0].data.cpu().numpy().shape[2:])
                tumor_resized_to_third_conv_val = resize_tumor_to_feature_map(
                post_tumor_val_batch, third_conv_val[0].data.cpu().numpy().shape[2:])
                
                val_loss_1 = criterion(first_conv_val[0], first_conv_val[1], tumor_resized_to_first_conv_val)
                val_loss_2 = criterion(second_conv_val[0], second_conv_val[1], tumor_resized_to_second_conv_val)
                val_loss_3 = criterion(third_conv_val[0], third_conv_val[1], tumor_resized_to_third_conv_val)
                val_loss: torch.Tensor = val_loss_1 + val_loss_2 + val_loss_3
                # np.save(os.path.join(os.getcwd(), "pre_val.npy"), third_conv_val[0].data.cpu().numpy())
                # np.save(os.path.join(os.getcwd(), "post_val.npy"), third_conv_val[1].data.cpu().numpy())
                epoch_val_loss += val_loss.item()      

                ###
                ##################################################################################    
                batch_f1_scores = 0.0
                for batch_index in range(pre_val_batch.size(0)):
                    #TODO: stopping criteria needs to be relaxed i think.. 
                    # Check loss for similar pairs?
                                 
                    distance_map_1 = return_upsampled_norm_distance_map(first_conv_val[0][batch_index], first_conv_val[1][batch_index],
                                                                dist_flag=dist_flag, mode='bilinear')
                    distance_map_2 = return_upsampled_norm_distance_map(second_conv_val[0][batch_index], second_conv_val[1][batch_index],
                                                                dist_flag=dist_flag, mode='bilinear')
                    distance_map_3 = return_upsampled_norm_distance_map(third_conv_val[0][batch_index], third_conv_val[1][batch_index],
                                                                dist_flag=dist_flag, mode='bilinear')
                    f1_score1, _, _, _ = eval_feature_map(post_tumor_val_batch.cpu().numpy()[batch_index][0], distance_map_1, 0.30, 
                                                  beta=1)
                    f1_score2, _, _, _ = eval_feature_map(post_tumor_val_batch.cpu().numpy()[batch_index][0], distance_map_2, 0.30, 
                                                   beta=1)
                    f1_score3, _, _, _ = eval_feature_map(post_tumor_val_batch.cpu().numpy()[batch_index][0], distance_map_3, 0.30, 
                                                            beta=1)
                    batch_f1_scores += (f1_score1 + f1_score2 + f1_score3) / 3
                batch_f1_scores /= pre_val_batch.size(0) 
                epoch_f1_scores += batch_f1_scores        
        # Calculate average loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_f1_score = epoch_f1_scores / len(val_loader)
        
        #print(f"Average sample loss for epoch {epoch+1}: Train Loss: {epoch_train_loss/total_train_samples}, Val Loss: {epoch_val_loss/total_val_samples}")
        print(f'Epoch [{epoch+1}/{epochs}], Average Train Loss: {avg_train_loss:.4f}, Average val loss: {avg_val_loss:.4f},\
              Average f1 score {avg_f1_score:.4f}')
        
        # Check for improvement in validation loss
        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
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
        criterion = contrastiveThresholdMaskLoss(hingethresh=args.threshold, margin=args.margin,
                                                 dist_flag=args.dist_flag)
        transform = Compose([
                    T.ToTensor(), # First to tensor then do tensor-based transformations
                    ShiftImage(max_shift_x=50, max_shift_y=50),
                    # T.RandomVerticalFlip(),
                    # T.RandomHorizontalFlip(),
                    RotateImage(padding_mode='border', align_corners=True)
                    ]
        )
            
    # aertsImages = aertsDataset(proc_preop=args.aerts_dir, 
    #             raw_tumor_dir=args.tumor_dir, save_dir=args.slice_dir,
    #             image_ids=['t1_ants_aligned.nii.gz'], skip=args.skip, 
    #             tumor_sensitivity=0.30,transform=transform, load_slices=args.load_slices)
    # print("Aerts dataset loaded")
    remindImages = remindDataset(preop_dir=args.remind_dir, 
                image_ids=['t1_aligned_stripped'], save_dir=args.slice_dir,
                skip=args.skip, tumor_sensitivity=0.30, transform=transform, load_slices=args.load_slices)
    # subject_images = ConcatDataset([aertsImages, remindImages])
    subject_images = remindImages
    from network import DeepLabV3
    model_type = DeepLabV3()
    # balance subject_images based on label
    
    print(f"Total number of images: {len(subject_images)}")
    subject_images: list[dict] = balance_dataset(subject_images)
    print(f"Total number of total pairs after balancing: {len(subject_images)}")
    train_subject_images, val_subject_images, test_subject_images = random_split(subject_images, (0.8, 0.1, 0.1))
    
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
            save_dir=save_dir, device=device, dist_flag=args.dist_flag)

    distances, labels, f_score, miou, all_f1_scores = predict(model_type, test_loader, base_dir =save_dir, device=device, dist_flag=args.dist_flag)

    # take the conv distance distance from each tuple
    thresholds = generate_roc_curve([d[0].item() for d in distances], labels, save_dir, f"_conv1_{f_score}_miou_{miou}")
    thresholds = generate_roc_curve([d[1].item() for d in distances], labels, save_dir, f"_conv2_{f_score}_miou_{miou}")
    thresholds = generate_roc_curve([d[2].item() for d in distances], labels, save_dir, f"_conv3_{f_score}_miou_{miou}")
    create_histogram_f1(all_f1_scores, save_dir)
    