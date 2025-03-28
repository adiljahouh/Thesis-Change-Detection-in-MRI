import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

class contrastiveLoss(nn.Module):

    def __init__(self,margin =2.0,dist_flag='l2'):
        super(contrastiveLoss, self).__init__()
        self.margin = margin
        self.dist_flag = dist_flag

    def various_distance(self,out_vec_t0,out_vec_t1):

        if self.dist_flag == 'l2': # Euclidean distance
            distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
        if self.dist_flag == 'l1': # Manhattan distance
            distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=1)
        if self.dist_flag == 'cos':# Cosine similarity
            similarity = F.cosine_similarity(out_vec_t0,out_vec_t1)
            distance = 1 - 2 * similarity/np.pi
        return distance

    def forward(self,out_vec_t0,out_vec_t1,label) -> torch.Tensor:
        out_vec_t0 = out_vec_t0.view(out_vec_t0.size(0), -1)  
        out_vec_t1 = out_vec_t1.view(out_vec_t1.size(0), -1)
        
        distance = self.various_distance(out_vec_t0,out_vec_t1)
        #constractive_loss = (label)*torch.pow(distance,2 ) + \
        #                               1-label * torch.pow(torch.clamp(self.margin - distance, min=0.0),2)
        loss = (label * torch.pow(distance, 2) +
                (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return torch.mean(loss)  
        ## if 1 (simillar) constractive loss =  distance^2
        ## if 0 (dissimilar) constractive loss = ((margin - distance)^2)
    
class contrastiveThresholdLoss(nn.Module):

    """
    This is a loss function that is used to train a siamese network
    The loss function is a combination of a threshold and a margin
    
    It does a loss function over the ENTIRE batch on LABEL basis NOT pixel basis (so no masks)
    we expect two tensors and a label to be passed to the forward function
    
    the distance map IS NOT pixel-wise, it just cares for the distance between the two vectors
    so thats why its a bit agnostic
    """
    def __init__(self,hingethresh=0.0,margin=2.0, dist_flag='l2'):
        super(contrastiveThresholdLoss, self).__init__()
        self.threshold = hingethresh
        self.margin = margin
        self.dist_flag = dist_flag

    def various_distance(self,out_vec_t0,out_vec_t1):

        if self.dist_flag == 'l2': # Euclidean distance
            distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
        if self.dist_flag == 'l1': # Manhattan distance
            distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=1)
        if self.dist_flag == 'cos':# Cosine similarity
            similarity = F.cosine_similarity(out_vec_t0,out_vec_t1)
            distance = 1 - 2 * similarity/np.pi
        return distance
    
    def forward(self,out_vec_t0,out_vec_t1,label) -> torch.Tensor:
        # TODO: FLATTENS TO 1D to just care about the distance between the two vectors
        # pixel-based distance not important for an agnostic model
        out_vec_t0 = out_vec_t0.view(out_vec_t0.size(0), -1)  
        out_vec_t1 = out_vec_t1.view(out_vec_t1.size(0), -1)
        
        distance = self.various_distance(out_vec_t0,out_vec_t1) # returns scalar distance (16,) since batch size is 16
        
        ## in my code the margin is used for dissimilar pairs 0
        ## and the threshold is used for similar pairs 1
        ## This might not be the regular usage of margin and threshold
        similar_pair = torch.clamp(distance - self.threshold,min=0.0)  # dont require similar pairs to overfit and be loss of 0
        dissimilar_pair = torch.clamp(self.margin- distance,min=0.0) # push dissimalar pairs to  be greater than margin
        #dissimilar_pair = torch.clamp(self.margin-(distance-self.threshold),min=0.0)
        constractive_thresh_loss =  label* torch.pow(similar_pair,2) + (1-label) * torch.pow(dissimilar_pair,2)
        # if 1 (simillar) constractive loss = margin - distance
        return torch.mean(constractive_thresh_loss)

def resize_tumor_to_feature_map(label, size) -> torch.Tensor:
    interp = nn.Upsample(size=size,mode='bilinear')
    return interp(label)
class contrastiveThresholdMaskLoss(nn.Module):
    """
    This is a loss function that is used to train a siamese network
    The loss function is a combination of a threshold and a margin
    
    It does a loss function on pixel basis so the label is depended on the pixel
    so we focus directly on if the featuremap overlays the ground truth tumor
    """
    def __init__(self,hingethresh,margin, dist_flag):
        super(contrastiveThresholdMaskLoss, self).__init__()
        self.threshold = hingethresh
        self.margin = margin
        self.dist_flag = dist_flag

    def various_distance(self,out_vec_t0,out_vec_t1):

        if self.dist_flag == 'l2': # Euclidean distance
            distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
        if self.dist_flag == 'l1': # Manhattan distance
            distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=1)
        if self.dist_flag == 'cos':# Cosine similarity
            similarity = F.cosine_similarity(out_vec_t0,out_vec_t1)
            distance = 1 - 2 * similarity/np.pi
        return distance
    
    def forward(self,map_t0, map_t1,ground_truth):
        n, c, h, w = map_t0.shape
    
        # Flatten the spatial dimensions
        out_t0_rz = map_t0.view(n, c, -1).transpose(1, 2)  # Shape: (n, h*w, c)
        out_t1_rz = map_t1.view(n, c, -1).transpose(1, 2)  # Shape: (n, h*w, c)
        # Calculate pairwise distance
        distance = self.various_distance(out_t0_rz, out_t1_rz)  # Shape: (n, h*w), basically distance scalar for each pixel vector (over all channels)
        # print(f"Shape of distance: {distance.shape}")
        # print(f"Shape of map_t0: {map_t0.shape}")
        # print(f"Shape of map_t1: {map_t1.shape}")
        # print(f"Shape of ground_truth: {ground_truth.shape}")
        # Reshape distance to match the ground truth shape
        distance = distance.view(n, h, w)  # Shape: (n, h, w) reshape it back to the original shape with distance scalar per pixel
        # Ensure ground truth tensor is compatible
        gt_rz = ground_truth.squeeze(1)  # Shape: (n, h, w)

        # if torch.sum(gt_rz) == 0:
        #      similar_pair_penalty = torch.clamp(distance, min=0.0)
        # else:
        # # Calculate the contrastive threshold loss
        similar_pair_penalty = torch.clamp(distance - self.threshold, min=0.0) # distance below threshold resolves to 0
        dissimilar_pair_penalty = torch.clamp(self.margin - distance, min=0.0) # distance above margin resolves to 0

        #print(f"Similar pair penalty min: {similar_pair_penalty.min()}, max: {similar_pair_penalty.max()}")
        ## similar part, multiply NON tumor area intensity (by inversion)
        ## with the similar pair penalty (clipped at thresh), so we only get the loss for the NON
        # tumor area
        
        ## vice versa for the dissimilar part
        ## loss += tumor area * distance map clipped at margin
        ## loss += NON tumor area * distance map clipped at threshold
        
        constractive_thresh_loss = torch.mean(
            (1 - gt_rz) * torch.pow(similar_pair_penalty, 2) + gt_rz * torch.pow(dissimilar_pair_penalty, 2)
        )
        
        #attempting to unsupervise the tumor area
        # constractive_thresh_loss = torch.sum(
        #     (1 - gt_rz) * torch.pow(similar_pair_penalty, 2)
        # )
        return constractive_thresh_loss

import numpy as np

def compute_iou(tumor_seg, feature_map=None, threshold=None, pred_mask=None):
    """
    Computes the Intersection over Union (IoU) for a given threshold or a precomputed segmentation mask.
    
    Parameters:
    - tumor_seg: Ground truth binary segmentation (2D numpy array, 1 = tumor, 0 = background).
    - feature_map: Model output (2D numpy array with probability values between 0 and 1) [optional if pred_mask is given].
    - threshold: The threshold to binarize the feature map [required if pred_mask is not given].
    - pred_mask: Precomputed binary segmentation mask (2D numpy array, 1 = tumor, 0 = background) [optional].
    
    Returns:
    - IoU score for the given threshold or mask.
    """
    if pred_mask is None:
        if feature_map is None or threshold is None:
            raise ValueError("Either pred_mask or both feature_map and threshold must be provided.")
        # Convert feature map into binary mask using the given threshold
        pred_mask = feature_map >= threshold  # 1 where detected as tumor, 0 otherwise

    # True Positives (TP): Pixels correctly classified as tumor
    TP = np.sum((pred_mask == 1) & (tumor_seg == 1))

    # False Positives (FP): Pixels incorrectly classified as tumor
    FP = np.sum((pred_mask == 1) & (tumor_seg == 0))

    # False Negatives (FN): Tumor pixels that were missed
    FN = np.sum((pred_mask == 0) & (tumor_seg == 1))

    # Compute IoU
    IoU = TP / (TP + FP + FN + 1e-10)  # Avoid division by zero

    return IoU


def compute_f_score(tumor_seg, feature_map=None, threshold=None, pred_mask=None, beta=0.8):
    """
    Computes the F-score for a given threshold or a precomputed segmentation mask.
    
    Parameters:
    - tumor_seg: Ground truth binary segmentation (2D numpy array, 1 = tumor, 0 = background).
    - feature_map: Model output (2D numpy array with probability values between 0 and 1) [optional if pred_mask is given].
    - threshold: The threshold to binarize the feature map [required if pred_mask is not given].
    - pred_mask: Precomputed binary segmentation mask (2D numpy array, 1 = tumor, 0 = background) [optional].
    - beta: Beta value for the F-score (default = 0.8, lower values emphasize precision).
    
    Returns:
    - F-score for the given threshold or mask.
    - Precision and recall values.
    """
    if pred_mask is None:
        if feature_map is None or threshold is None:
            raise ValueError("Either pred_mask or both feature_map and threshold must be provided.")
        # Convert feature map into binary mask using the given threshold
        pred_mask = feature_map >= threshold  # 1 where detected as tumor, 0 otherwise

    # True Positives (TP): Pixels correctly classified as tumor
    TP = np.sum((pred_mask == 1) & (tumor_seg == 1))

    # False Positives (FP): Pixels incorrectly classified as tumor
    FP = np.sum((pred_mask == 1) & (tumor_seg == 0))

    # False Negatives (FN): Tumor pixels that were missed
    FN = np.sum((pred_mask == 0) & (tumor_seg == 1))

    # Compute precision and recall
    precision = TP / (TP + FP + 1e-10)  # Add small value to avoid division by zero
    recall = TP / (TP + FN + 1e-10)

    # Compute F-score
    betasq = beta ** 2
    f_score = (1 + betasq) * (precision * recall) / ((betasq * precision) + recall + 1e-10)

    return f_score, precision, recall
   
def eval_feature_map(tumor_seg, feature_map, seg_value_index, beta=0.8):
    """
       tumor seg is the ground rtuth
        prob is the binary image feature map
        seg_value_index: what do we consider tumor at what thresh
        beta is the for the f1 score, lower = more emphasis on reducing false positives
        since we are looking for partial changes maybe we should use a lower beta
    """
    # randint = np.random.randint(0, 1000)
    thresh = np.array(range(0, 256))/255.0 
    significant_tumor_pixels = tumor_seg[:,:] > seg_value_index ## 0.30 check for tumor pixels
    all_tumor_pixels = tumor_seg[:, :] != 0 # full segmentation area
    
     
    FN, FP, posNum, negNum = calc_fn_fp_per_thresh(all_tumor_pixels, feature_map,
                                     thresh)
    best_f1, f1_thresh, thresh_index, precision, recall = find_best_thresh_for_f1(FN, FP, posNum, thresh, beta=beta)
    best_miou = find_best_thresh_for_miou(FN, FP, posNum, thresh_index)
    return best_f1, best_miou, precision, recall, 

def find_best_thresh_for_f1(FN, FP, posNum, thresh, beta=0.8):
    # Calculate precision, recall, and beta-weighted F-score for each threshold
    tp = posNum - FN  # True positives at each threshold
    precision = tp / (tp + FP + 1e-10)  # Avoid division by zero
    recall = tp / (posNum + 1e-10)  # Avoid division by zero

    betasq = beta**2
    F = (1 + betasq) * (precision * recall) / ((betasq * precision) + recall + 1e-10)

    # Find the best threshold based on F-score
    best_index = F.argmax()
    best_f1 = F[best_index]
    best_threshold = thresh[best_index]
    return best_f1, best_threshold, best_index, precision[best_index], recall[best_index]

def find_best_thresh_for_miou(FN, FP, posNum, thresh_index):
    """
    Find the threshold that maximizes the Mean IoU (mIoU) score.
    """
    TP = posNum - FN  # True positives at each threshold
    IoU = TP / (TP + FP + FN + 1e-10)  # Compute IoU for each threshold

    # Find the best threshold based on IoU score
    best_miou = IoU[thresh_index]
    return best_miou

def calc_fn_fp_per_thresh(significant_tumor_pixels, feature_map, thres):
    '''
    Does the basic pixel based evaluation!
    :param truthy_tumor_pixels: pixels over the threshold we use to tag images
    :param feature_map: the base feature map
    :param thres: array going up to 1
    :param validMap:
    :param segmentation_area: boolean mask where its not 255 = true
    '''

    assert len(feature_map.shape) == 2, f'Wrong size of input prob map {feature_map.shape}'
    assert len(significant_tumor_pixels.shape) == 2, f'Wrong size of input prob map {feature_map.shape}'
    thresInf = np.concatenate(([-np.Inf], thres, [np.Inf])) 
    fnArray = feature_map[(significant_tumor_pixels == True)] # pixels in fm where tumor is located
    ## array of probabilities of tumor pixels [0.8, 0.9, 0.7, 0.6, 0.5]
    fnHist = np.histogram(fnArray,bins=thresInf)[0]
    ## turns it into counts per bins(thresholds!) so for bin 0.0 we have 0 FN, 
    # for bin 1.0 we need to sum all bins
    # and then we see that everything is a FN
    
    
    ## we put them in bins to see at what threshold they are detected as true positives in fm
    fnCum = np.cumsum(fnHist)
    ## cumsum makes converts the bins to count the amount of FN at each threshold
    FN = fnCum[0:0+len(thres)]
    
    fpArray = feature_map[(significant_tumor_pixels == False)] # pixels in fm where tumor is not located
    
    fpHist  = np.histogram(fpArray, bins=thresInf)[0]
    fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))
    FP = fpCum[1:1+len(thres)]

    posNum = np.sum(significant_tumor_pixels == True)
    negNum = np.sum(significant_tumor_pixels == False)
    ## return the number of false negatives, false positives per threshold
    # number of positive pixels, number of negative pixels
    return FN, FP, posNum, negNum
    