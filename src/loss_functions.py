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



def eval_feature_map(gt_image, prob,cl_index):
    """
        gt image is the ground truth image
        prob is the binary image feature map
        cl_index is set to 1?
    """
    thresh = np.array(range(0, 256))/255.0 
    cl_gt = gt_image[:,:] == cl_index ## Makes boolean map if the value is 1
    valid_gt = gt_image[:,:] != 255 # makes boolean map if the value is not 255

    FN, FP, posNum, negNum = evalExp(cl_gt, prob,
                                     thresh, validMap=None,
                                     validArea=valid_gt)
    return FN, FP, posNum, negNum

def evalExp(gtBin, cur_prob, thres, validMap = None, validArea=None):
    '''
    Does the basic pixel based evaluation!
    :param gtBin: Boolean mask if the pixel is 1 = true in ground truth
    :param cur_prob: the base feature map
    :param thres: array going up to 1
    :param validMap:
    :param validArea: boolean mask where its not 255 = true
    '''

    assert len(cur_prob.shape) == 2, 'Wrong size of input prob map'
    assert len(gtBin.shape) == 2, 'Wrong size of input prob map'
    thresInf = np.concatenate(([-np.Inf], thres, [np.Inf]))
    
    #Merge validMap with validArea
    if np.any(validMap)!=None:
        if np.any(validArea)!=None:
            validMap = (validMap == True) & (validArea == True)
    elif np.any(validArea)!=None:
        validMap=validArea

    # histogram of false negatives
    if np.any(validMap)!=None:
        #valid_array = cur_prob[(validMap == False)]
        fnArray = cur_prob[(gtBin == True) & (validMap == True)]
    else:
        fnArray = cur_prob[(gtBin == True)]
    #f = np.histogram(fnArray,bins=thresInf)
    fnHist = np.histogram(fnArray,bins=thresInf)[0]
    fn_list = list(fnHist)
    fnCum = np.cumsum(fnHist)
    FN = fnCum[0:0+len(thres)]
    
    if validMap.any()!=None:
        fpArray = cur_prob[(gtBin == False) & (validMap == True)]
    else:
        fpArray = cur_prob[(gtBin == False)]
    
    fpHist  = np.histogram(fpArray, bins=thresInf)[0]
    fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))
    FP = fpCum[1:1+len(thres)]

    # count labels and protos
    #posNum = fnArray.shape[0]
    #negNum = fpArray.shape[0]
    if np.any(validMap)!=None:
        posNum = np.sum((gtBin == True) & (validMap == True))
        negNum = np.sum((gtBin == False) & (validMap == True))
    else:
        posNum = np.sum(gtBin == True)
        negNum = np.sum(gtBin == False)
    return FN, FP, posNum, negNum
