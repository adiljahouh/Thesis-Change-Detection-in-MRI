import torch.nn as nn
import torch
import numpy as np
import cv2
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from distance_measures import various_distance
def merge_images(pre, post, heatmap, baseline, output_path):
    # Create a new figure
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))

    # Display each image on a separate subplot
    axs[0].imshow(pre, cmap="gray")
    axs[0].axis('off')
    axs[1].imshow(post, cmap="gray")
    axs[1].axis('off')
    axs[2].imshow(heatmap, cmap="jet")
    axs[2].axis('off')
    axs[3].imshow(baseline, cmap="jet")
    axs[3].axis('off')
    
    # # Create a colorbar
    # cbar = plt.colorbar(im, ax=axs[2], orientation='vertical')
    # cbar.set_label('Similarity Distance (Scaled by 255)')
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save the merged image
    plt.savefig(output_path)
    plt.close(fig)

def get_baseline(pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:  
    diff = torch.abs(pre - post)
    return diff.data.cpu().numpy()

def generate_roc_curve(distances, labels, save_dir):
    # # Invert distances because lower distance indicates more similarity
    # distances = [d.cpu().item() for d in distances]
    # labels = [l.cpu().item() for l in labels]

    scores = [-d for d in distances]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the plot to the specified directory
    plt.savefig(os.path.join(save_dir, f'roc.png'))
    plt.close()  # Close the plot to free up memory
    return thresholds

def single_layer_similar_heatmap_visual(output_t0: torch.Tensor,output_t1: torch.Tensor,dist_flag: str):

    interp = nn.Upsample(size=[256,256], mode='bilinear')
    c, h, w = output_t0.data.shape
    print("shape: ", c, h, w)
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    # print(distance)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()
    print("similar_distance_map: ", similar_distance_map.shape)
    ## create a 4 dim torch by adding 2 axis to h,w
    ## torch upsamle expects b,c,h,w
    similar_distance_map_rz = interp(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :]))
    # print(similar_distance_map_rz)
    similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    return similar_dis_map_colorize, similar_distance_map_rz.data.cpu().numpy()[0][0]

def plot_and_save_ndarray(data, save_dir, filename):
    # Create a new figure
    plt.figure()

    # Display the data as an image
    plt.imshow(data, cmap='gray')

    # Remove axes
    plt.axis('off')

    # Save the plot to the specified directory
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()  # Close the plot to free up memory
    