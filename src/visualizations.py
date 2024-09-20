import torch.nn as nn
import torch
import numpy as np
import cv2
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from distance_measures import various_distance
from loader import normalize_np_array
def merge_images(*args, output_path, title, **kwargs):
    """
    Merges and visualizes a variable number of images in a single figure.
    
    Args:
        *args: Variable number of image arrays (2D or 3D).
        output_path (str): Path where the merged image will be saved.
        **kwargs: Additional keyword arguments (currently not used but can be extended).
    """
    # Number of images
    num_images = len(args)
    
    # Create a figure with a number of subplots equal to the number of images
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 3, 4))
    
    # If only one subplot is created, axs will not be an array
    if num_images == 1:
        axs = [axs]
    
    # Display each image on a separate subplot
    for i, img in enumerate(args):
        if i < 2:
            axs[i].imshow(img, cmap="gray")  # First two images use grayscale colormap
        else:
            axs[i].imshow(img, cmap="jet")   # Subsequent images use jet colormap
        axs[i].axis('off')
    
    # Adjust spacing between subplots
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the merged image
    plt.savefig(output_path)
    plt.close(fig)

# Example usage

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

def single_layer_similar_heatmap_visual(output_t0: torch.Tensor,output_t1: torch.Tensor,dist_flag: str,
                                        mode='bilinear'):

    interp = nn.Upsample(size=[256,256], mode=mode)
    c, h, w = output_t0.data.shape
    # print("shape: ", c, h, w)
    # TODO: check that hte loss is normalized
    # TODO: Create pre pre pairs to make sure output is correct
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()
    # print("similar_distance_map: ", similar_distance_map.shape)

    ## create a 4 dim torch by adding 2 axis to h,w
    ## torch upsamle expects b,c,h,w
    ## normalize it after to 0 1
    similar_distance_map_rz = interp(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :]))
    normalized_distance_map = normalize_np_array(similar_distance_map_rz.data.cpu().numpy()[0][0])
    assert normalized_distance_map.max() <= 1.0, f"max: {normalized_distance_map.max()}"
    assert normalized_distance_map.min() >= 0.0, f"min: {normalized_distance_map.min()}"

    similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    return similar_dis_map_colorize, normalized_distance_map

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
    