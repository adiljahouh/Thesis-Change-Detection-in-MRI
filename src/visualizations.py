import torch
import numpy as np
import cv2
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from distance_measures import various_distance
from loader import normalize_np_array
from matplotlib.axes import Axes
from typing import Tuple

interp = torch.nn.Upsample(size=(256, 256), mode='bilinear')

def visualize_multiple_fmaps_and_tumor_baselines(*args: Tuple[np.ndarray, str], 
                                                 output_path: str, tumor: np.ndarray, 
                                                 pre_non_transform: np.ndarray, **kwargs):
    """
    Merges and visualizes a variable number of images in a single figure.
    
    Args:
        *args: Variable number of image arrays (2D or 3D).
        output_path (str): Path where the merged image will be saved.
        **kwargs: Additional keyword arguments (currently not used but can be extended).
    """
    tumor_overlay_normalized = None
    num_images = len(args) + 1
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
        axs[i].set_title(title)
        if i < 2:
            axs[i].imshow(img, cmap="gray")  # First two images use grayscale colormap
        else:
            axs[i].imshow(img, cmap="jet", alpha=1)   # Subsequent overlays use jet colormap
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

def visualize_multiple_images(*args: Tuple[np.ndarray, str], output_path: str, **kwargs):
    """
    Merges and visualizes a variable number of images in a single figure.
    
    Args:
        *args: Variable number of image arrays (2D or 3D).
        output_path (str): Path where the merged image will be saved.
        **kwargs: Additional keyword arguments (currently not used but can be extended).
    """
    num_images = len(args) // 2
    
    # Create a figure with a number of subplots equal to the number of image pairs
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 3, 4))
    axs: list[Axes]
    # If only one subplot is created, axs will not be an array
    if num_images == 1:
        axs = [axs]
    
    # Display each pair of images on a separate subplot
    for i in range(0, len(args), 2):
        base_img, base_title = args[i]
        overlay_img, overlay_title = args[i + 1]
        
        axs[i // 2].axis('off')
        axs[i // 2].set_title(f"{base_title} + {overlay_title}")
        axs[i // 2].imshow(base_img, cmap="gray")
        axs[i // 2].imshow(overlay_img, cmap="jet", alpha=0.5)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save the merged image
    plt.savefig(output_path)
    plt.close(fig)
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
def return_distance_map(output_t0: torch.Tensor,output_t1: torch.Tensor,dist_flag: str) -> torch.Tensor:
    c, h, w = output_t0.data.shape

    # remember the c, h, w -> flatten
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    return distance.view(h,w)
def return_upsampled_distance_map(output_t0: torch.Tensor,output_t1: torch.Tensor,dist_flag: str,
                                        mode='bilinear') -> torch.Tensor:

    # interp = torch.nn.Upsample(size=[256,256], mode=mode)
    c, h, w = output_t0.data.shape

    # remember the c, h, w -> flatten
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()
    similar_distance_map_rz = interp(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :]))
    return similar_distance_map_rz

def return_upsampled_norm_distance_map(output_t0: torch.Tensor,output_t1: torch.Tensor,dist_flag: str,
                                        mode='bilinear'):

    # interp = torch.nn.Upsample(size=[256,256], mode=mode)
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
    return normalized_distance_map

def multiplicative_sharpening_and_filter(distance_map: np.ndarray, base_image: np.ndarray, alpha=2.0, beta=1, threshold=0.15):
    """Apply multiplicative sharpening by injecting high-frequency details from the base image."""
    assert distance_map.max() <= 1.0, f"max: {distance_map.max()}"
    assert distance_map.min() >= 0.0, f"min: {distance_map.min()}"
    assert base_image.max() <= 1.0, f"max: {base_image.max()}"
    assert base_image.min() >= 0.0, f"min: {base_image.min()}"

    # Create a binary mask where base image values are above the threshold
    mask = (base_image > threshold).astype(np.float32)
    
    # Blur the base image and calculate high-frequency details
    blurred_image = cv2.GaussianBlur(base_image, (5, 5), 0)
    high_freq_details = base_image - blurred_image

    # Compute sharpened map for high-value regions
    try:
        sharpened_map = distance_map * (1 + alpha * high_freq_details)
        
        # Add contribution from the base image in low-value regions
        enhanced_map = sharpened_map + beta * base_image * (1 - distance_map)

        # Apply the mask to filter the distance map
        enhanced_map = mask * enhanced_map + (1 - mask) * base_image

        # Normalize the final result to keep it within [0, 1]
        norm_enhanced_map = normalize_np_array(enhanced_map)
    except ValueError as e:
        return distance_map
    print(f"enhanced_map: {enhanced_map.max()}, {enhanced_map.min()}")
    print(f"norm_enhanced_map: {norm_enhanced_map.max()}, {norm_enhanced_map.min()}")
    return norm_enhanced_map

def multiplicative_sharpening(distance_map: np.ndarray, base_image: np.ndarray, alpha=4.0, beta=0.5):
    """Apply multiplicative sharpening by injecting high-frequency details from the base image."""
    assert distance_map.max() <= 1.0, f"max: {distance_map.max()}"
    assert distance_map.min() >= 0.0, f"min: {distance_map.min()}"
    assert base_image.max() <= 1.0, f"max: {base_image.max()}"
    assert base_image.min() >= 0.0, f"min: {base_image.min()}"

    # Blur the base image and calculate high-frequency details
    blurred_image = cv2.GaussianBlur(base_image, (5, 5), 0)
    high_freq_details = base_image - blurred_image

    # Compute sharpened map for high-value regions
    sharpened_map = distance_map * (1 + alpha * high_freq_details)
    
    # Add contribution from the base image in low-value regions
    enhanced_map = sharpened_map + beta * base_image * (1 - distance_map)


    # Normalize the final result to keep it within [0, 1]
    enhanced_map = normalize_np_array(enhanced_map)
    
    return enhanced_map

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
    