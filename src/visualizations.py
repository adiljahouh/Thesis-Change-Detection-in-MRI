import torch.nn as nn
import torch
import numpy as np
import cv2
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from distance_measures import various_distance
from loader import normalize_np_array
from matplotlib.axes import Axes

def merge_and_overlay_images(*args, output_path, tumor, pre_non_transform, filter=True, **kwargs):
    """
    Merges and visualizes a variable number of images in a single figure.
    
    Args:
        *args: Variable number of image arrays (2D or 3D).
        output_path (str): Path where the merged image will be saved.
        **kwargs: Additional keyword arguments (currently not used but can be extended).
    """
    tumor_overlay_normalized = None
    num_images = len(args) + 2
    if tumor is not None:
        tumor_overlay = np.ma.masked_where(tumor == 0, tumor)
        tumor_overlay_normalized = tumor_overlay / np.max(tumor_overlay) if np.max(tumor_overlay) > 0 else tumor_overlay
    
    # Create a figure with a number of subplots equal to the number of images
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 3, 4))
    axs: list[Axes]
    # If only one subplot is created, axs will not be an array
    if num_images == 1:
        axs = [axs]
    
    # Display each image on a separate subplot
    for i, (img, title) in enumerate(args):
        axs[i].axis('off')
        if i < 2:
            axs[i].imshow(img, cmap="gray")  # First two images use grayscale colormap
            axs[i].set_title(title)
        else:
            if filter and "Baseline" not in title:
                # if conv layer, filter out noise
                img = np.ma.masked_where(img < 0.7, img)
                axs[i].imshow(args[0][0], cmap="gray") #preop underlay
            axs[i].imshow(img, cmap="jet", alpha=1, vmin=0, vmax=1)   # Subsequent overlays use jet colormap
            axs[i].set_title(title)
    axs[-2].imshow(args[2][0], cmap='jet')
    axs[-2].set_title('Distance Map conv 1')
    if tumor is not None:
        axs[-1].imshow(pre_non_transform, cmap='gray')
        axs[-1].imshow(tumor_overlay_normalized, cmap='jet', alpha=1) # tumor should be red
    else:
        blank_image = np.zeros_like(img, dtype=np.float32)  # Ensure the size matches other images
        axs[-1].imshow(blank_image, cmap="gray")
        axs[-1].text(
            0.5, 0.5, 
            "Image not available\n(Control slice)", 
            ha="center", va="center", 
            transform=axs[-1].transAxes, 
            fontsize=10, color="red"
        )
    axs[-1].axis('off')
    axs[-1].set_title('Tumor Segmentation')
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save the merged image
    plt.savefig(output_path)
    plt.close(fig)

# Example usage

def get_baseline_torch(pre: torch.Tensor, post: torch.Tensor) -> np.ndarray:  
    diff = torch.abs(pre - post)
    return diff.data.cpu().numpy()

def get_baseline_np(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    diff = np.abs(pre - post)
    return diff
def generate_roc_curve(distances, labels, save_dir, extra_title=""):
    # # Invert distances because lower distance indicates more similarity
    try:
        scores = [d.cpu().item() for d in distances]
        scores = [-s for s in scores]
    except AttributeError:
        scores = [-s for s in distances]
    # print(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Calculate Youden's J statistic for each threshold
    J_scores = tpr - fpr
    optimal_idx = np.argmax(J_scores)
    optimal_threshold = thresholds[optimal_idx]

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
    print("Saving ROC curve to:", os.path.join(save_dir, f'roc{extra_title}.png'))

    plt.savefig(os.path.join(save_dir, f'roc{extra_title}.png'))
    plt.close()  # Close the plot to free up memory
    print(optimal_threshold)
    return optimal_threshold

def return_upsampled_norm_distance_map(output_t0: torch.Tensor,output_t1: torch.Tensor,dist_flag: str,
                                        mode='bilinear'):
    interp = nn.Upsample(size=[256,256], mode=mode)
    c, h, w = output_t0.data.shape
    # remember the c, h, w -> flatten
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()
    ## create a 4 dim torch by adding back h, w axis post distance calc 
    
    ## torch upsamle expects b,c,h,w
    ## normalize it after to 0 1
    
    similar_distance_map_rz = interp(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :]))
    normalized_distance_map = normalize_np_array(similar_distance_map_rz.data.cpu().numpy()[0][0])
    try:
        assert normalized_distance_map.max() <= 1.0, f"max: {normalized_distance_map.max()}, {normalized_distance_map}, \n {similar_distance_map_rz.data.cpu().numpy()[0][0]}"
        assert normalized_distance_map.min() >= 0.0, f"min: {normalized_distance_map.min()}"
    except AssertionError as e:
        print(f"AssertionError caught: {e}")
    # Save the tensor for further debugging
        save_path = "./debug/similar_distance_map_rz_failed.txt"
        os.makedirs("./debug", exist_ok=True)
        np.savetxt(save_path, similar_distance_map)
    ## NOTE: I do not use the cv2, i just use the normalized_distance_map
    similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    return normalized_distance_map

def overlay_thresholded_dist_map_over_preop(preop: np.ndarray, dist_map: np.ndarray, threshold: float, save_dir: str, filename: str):
    pass
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
    