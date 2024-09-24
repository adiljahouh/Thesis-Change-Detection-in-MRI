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
        
        similar_pair = torch.clamp(distance - self.threshold,min=0.0)
        dissimilar_pair = torch.clamp(self.margin- distance,min=0.0)
        #dissimilar_pair = torch.clamp(self.margin-(distance-self.threshold),min=0.0)
        constractive_thresh_loss =  label* torch.pow(similar_pair,2) + (1-label) * torch.pow(dissimilar_pair,2)
        # if 1 (simillar) constractive loss = margin - distance
        return torch.mean(constractive_thresh_loss)