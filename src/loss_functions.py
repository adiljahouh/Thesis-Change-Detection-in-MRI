import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
class ConstractiveLoss(nn.Module):

    def __init__(self,margin =2.0,dist_flag='l2'):
        super(ConstractiveLoss, self).__init__()
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
    
class ConstractiveThresholdHingeLoss(nn.Module):

    def __init__(self,hingethresh=0.0,margin=2.0, dist_flag='l2'):
        super(ConstractiveThresholdHingeLoss, self).__init__()
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
        out_vec_t0 = out_vec_t0.view(out_vec_t0.size(0), -1)  
        out_vec_t1 = out_vec_t1.view(out_vec_t1.size(0), -1)
        
        distance = self.various_distance(out_vec_t0,out_vec_t1)
        ## in my code the margin is used for dissimilar pairs 0
        ## and the threshold is used for similar pairs 1
        ## This might not be the regular usage of margin and threshold
        similar_pair = torch.clamp(distance - self.threshold,min=0.0)  # dont require similar pairs to overfit and be loss of 0
        dissimilar_pair = torch.clamp(self.margin- distance,min=0.0) # push dissimalar pairs to  be greater than margin
        #dissimilar_pair = torch.clamp(self.margin-(distance-self.threshold),min=0.0)
        constractive_thresh_loss =  label* torch.pow(similar_pair,2) + (1-label) * torch.pow(dissimilar_pair,2)
        # if 1 (simillar) constractive loss = margin - distance
        return torch.mean(constractive_thresh_loss)



def eval_feature_map(tumor_seg, feature_map, seg_value_index, extra):
    """
        gt image is the ground truth image
        prob is the binary image feature map
        cl_index is set to 1?
    """
    # randint = np.random.randint(0, 1000)
    thresh = np.array(range(0, 256))/255.0 
    significant_tumor_pixels = tumor_seg[:,:] > seg_value_index ## 0.30 check for tumor pixels
    all_tumor_pixels = tumor_seg[:, :] != 0 # full segmentation area
    # has_tumor_pixels = np.any(all_tumor_pixels)
    # if not has_tumor_pixels:
    #     print(f"No tumor pixels found for {extra}")
    # # Visualize inputs
    # import matplotlib.pyplot as plt
    # import os
    # fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # axs[0].imshow(tumor_seg, cmap="gray")
    # axs[0].set_title(f"All tumor pixels (has tumor pixels: {has_tumor_pixels}) for {extra}")
    # axs[0].axis("off")
        
    # axs[1].imshow(significant_tumor_pixels, cmap="gray")
    # axs[1].set_title("significant tumor pixels")
    # axs[1].axis("off")
    
    # axs[2].imshow(feature_map, cmap="grey")
    # axs[2].set_title("Feature Map")
    # axs[2].axis("off")
    
    # # Save visualizations
    # vis_path = os.path.join("/home/adil/Documents/TUE/preparationPhase/myProject/src/tests", f"{randint}.png")
    # plt.tight_layout()
    # plt.savefig(vis_path, bbox_inches="tight")
    # plt.close(fig)
    


    FN, FP, posNum, negNum = evalExp(significant_tumor_pixels, feature_map,
                                     thresh, validMap=None,
                                     segmentation_area=all_tumor_pixels)
    return FN, FP, posNum, negNum

def evalExp(significant_tumor_pixels, feature_map, thres, validMap = None, segmentation_area=None):
    '''
    Does the basic pixel based evaluation!
    :param truthy_tumor_pixels: pixels over the threshold we use to tag images
    :param feature_map: the base feature map
    :param thres: array going up to 1
    :param validMap:
    :param segmentation_area: boolean mask where its not 255 = true
    '''

    assert len(feature_map.shape) == 2, 'Wrong size of input prob map'
    assert len(significant_tumor_pixels.shape) == 2, 'Wrong size of input prob map'
    thresInf = np.concatenate(([-np.Inf], thres, [np.Inf]))
    
    if np.any(segmentation_area)!=None:
        validMap=segmentation_area
        
    if np.any(validMap)!=None:
        fnArray = feature_map[(significant_tumor_pixels == True) & (validMap == True)]
    else:
        fnArray = feature_map[(significant_tumor_pixels == True)]
    #f = np.histogram(fnArray,bins=thresInf)
    fnHist = np.histogram(fnArray,bins=thresInf)[0]

    fnCum = np.cumsum(fnHist)
    FN = fnCum[0:0+len(thres)]
    
    if validMap.any()!=None:
        fpArray = feature_map[(significant_tumor_pixels == False) & (validMap == True)]
    else:
        fpArray = feature_map[(significant_tumor_pixels == False)]
    
    fpHist  = np.histogram(fpArray, bins=thresInf)[0]
    fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))
    FP = fpCum[1:1+len(thres)]

    # count labels and protos
    if np.any(validMap)!=None:
        posNum = np.sum((significant_tumor_pixels == True) & (validMap == True))
        negNum = np.sum((significant_tumor_pixels == False) & (validMap == True))
    else:
        posNum = np.sum(significant_tumor_pixels == True)
        negNum = np.sum(significant_tumor_pixels == False)
    return FN, FP, posNum, negNum
