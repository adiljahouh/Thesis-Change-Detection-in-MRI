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

    def forward(self,out_vec_t0,out_vec_t1,label):
        distance = self.various_distance(out_vec_t0,out_vec_t1)
        #constractive_loss = (label)*torch.pow(distance,2 ) + \
        #                               1-label * torch.pow(torch.clamp(self.margin - distance, min=0.0),2)
        
        loss = (label * torch.pow(distance, 2) +
                (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return torch.sum(loss)  # Ensure loss is a scalar
        ## contrastive loss = sum((1-label) * distance^2 + label * max(margin - distance,0)^2)
        ## if 1 (simillar) constractive loss = margin - distance
        ## if 0 (dissimilar) constractive loss = ((margin - distance)^2)
    
class ConstractiveThresholdHingeLoss(nn.Module):

    def __init__(self,hingethresh=0.0,margin=2.0):
        super(ConstractiveThresholdHingeLoss, self).__init__()
        self.threshold = hingethresh
        self.margin = margin

    def forward(self,out_vec_t0,out_vec_t1,label):

        distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
        similar_pair = torch.clamp(distance - self.threshold,min=0.0)
        dissimilar_pair = torch.clamp(self.margin- distance,min=0.0)
        #dissimilar_pair = torch.clamp(self.margin-(distance-self.threshold),min=0.0)
        constractive_thresh_loss = torch.sum(
            (1-label)* torch.pow(similar_pair,2) + label * torch.pow(dissimilar_pair,2)
        )
        return constractive_thresh_loss