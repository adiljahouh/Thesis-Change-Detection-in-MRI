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
def robust_normalize(arr, lower_percentile=0, upper_percentile=10):
    """Normalize array while ignoring extreme outliers."""
    vmin, vmax = np.percentile(arr, [lower_percentile, upper_percentile])  # Get stable min/max
    if vmax - vmin < 1e-8:  # Avoid divide-by-zero if array is nearly constant
        return np.zeros_like(arr)
    
    arr_clipped = np.clip(arr, vmin, vmax)  # Remove extreme values
    arr_norm = (arr_clipped - vmin) / (vmax - vmin)  # Normalize to [0,1]
    
    return arr_norm
def visualize_change_detection_control(
    *args: Tuple[np.ndarray, str, str, np.ndarray | None],  # change_map, title, scores, segmentation (possible)
    preoperative: Tuple[np.ndarray, np.ndarray], 
    postoperative: Tuple[np.ndarray, np.ndarray],
    ground_truth: Tuple[np.ndarray, np.ndarray] | None,
    output_path: str,
    show_gt: bool = False
):
    """
    Visualizes preoperative and postoperative images with overlays, followed by multiple change maps,
    and then overlays these change maps onto the postoperative image. The final column overlays the ground truth.

    Args:
        preoperative (Tuple[np.ndarray, np.ndarray]): (Grayscale preoperative image, Overlay mask).
        postoperative (Tuple[np.ndarray, np.ndarray]): (Grayscale postoperative image, Overlay mask).
        *args (Tuple[np.ndarray, str, str, np.ndarray]): Arbitrary number of change maps with labels. Possible segmentation mask.
        ground_truth (Tuple[np.ndarray, np.ndarray], optional): (Grayscale GT image, GT mask).
        output_path (str): Path to save the visualization.
        show_gt (bool): Whether to visualize the ground truth as a jet map separately.
    """
    num_maps = len(args)
    num_cols = 2 + num_maps * 2 + 1 + (1 if show_gt else 0)  # Adjust column count if show_gt is True
    
    fig, axs = plt.subplots(1, num_cols, figsize=(num_cols * 3, 4))
    
    # Unpack grayscale images and overlays
    preoperative_img, preoperative_overlay = preoperative
    postoperative_img, postoperative_overlay = postoperative

    # Display Preoperative Image with Overlay
    axs[0].imshow(preoperative_img, cmap="gray")
    axs[0].imshow(np.ma.masked_where(preoperative_overlay == 0, preoperative_overlay), cmap="jet", alpha=0.5)
    axs[0].axis("off")
    axs[0].set_title("Preoperative State")

    # Display Postoperative Image with Overlay
    axs[1].imshow(postoperative_img, cmap="gray")
    axs[1].imshow(np.ma.masked_where(postoperative_overlay == 0, postoperative_overlay), cmap="jet", alpha=0.5)
    axs[1].axis("off")
    axs[1].set_title("Intraoperative State")

    # Display Change Maps and Overlays
    for i, (change_map, title, score, seg_mask) in enumerate(args):
        idx = 2 + i  # Shift index after preoperative & postoperative images
        if "RiA" in title:
            change_map_mask = np.ma.masked_where(seg_mask == 0, seg_mask)
            alpha = 1
        else:
            change_map_norm = change_map / np.max(change_map) if np.max(change_map) > 0 else change_map
            change_map_mask = np.ma.masked_where(change_map_norm == 0, change_map_norm)
            alpha = 1

        # Change Map Visualization (Jet Colormap)
        axs[idx].imshow(change_map, cmap="jet", alpha=1)
        axs[idx].axis("off")
        axs[idx].set_title(title + f"\n{score})")
        
        # Overlay Change Map on Postoperative Image
        idx_overlay = 2 + num_maps + i  # Position after all change maps
        axs[idx_overlay].imshow(postoperative_img, cmap="gray")
        axs[idx_overlay].imshow(change_map_mask, cmap="jet", alpha=alpha)  # Overlay with transparency
        axs[idx_overlay].axis("off")
        axs[idx_overlay].set_title(f"{title}\n(superimposed)")
    
    # Display Ground Truth (if available and show_gt=True)
    if show_gt and ground_truth is not None:
        gt_img, gt_overlay = ground_truth
        gt_idx = -2  # Second to last column
        axs[gt_idx].imshow(gt_overlay, cmap="jet")
        axs[gt_idx].axis("off")
        axs[gt_idx].set_title("Ground Truth Mask $\Delta T$")

    # Display Final Ground Truth Overlay (if available)
    if ground_truth is not None:
        gt_img, gt_overlay = ground_truth
        axs[-1].imshow(gt_img, cmap="gray")
        axs[-1].imshow(np.ma.masked_where(gt_overlay == 0, gt_overlay), cmap="jet", alpha=0.5, vmin=0.7)
        axs[-1].axis("off")
        axs[-1].set_title("Ground Truth")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def visualize_change_detection(
    *args: Tuple[np.ndarray, str, str, np.ndarray | None],  # change_map, title, scores, segmentation (possible)
    preoperative: Tuple[np.ndarray, np.ndarray], 
    postoperative: Tuple[np.ndarray, np.ndarray],
    ground_truth: Tuple[np.ndarray, np.ndarray],
    output_path: str,
    show_gt: bool = False
):
    """
    Visualizes preoperative and postoperative images with overlays, followed by multiple change maps,
    and then overlays these change maps onto the postoperative image. The final column overlays the ground truth.

    Args:
        preoperative (Tuple[np.ndarray, np.ndarray]): (Grayscale preoperative image, Overlay mask).
        postoperative (Tuple[np.ndarray, np.ndarray]): (Grayscale postoperative image, Overlay mask).
        *args (Tuple[np.ndarray, str, str, np.ndarray]): Arbitrary number of change maps with labels. Possible segmentation mask.
        ground_truth (Tuple[np.ndarray, np.ndarray], optional): (Grayscale GT image, GT mask).
        output_path (str): Path to save the visualization.
        show_gt (bool): Whether to visualize the ground truth as a jet map separately.
    """
    num_maps = len(args)
    num_cols = 2 + num_maps * 2 + 1 + (1 if show_gt else 0)  # Adjust column count if show_gt is True
    
    fig, axs = plt.subplots(1, num_cols, figsize=(num_cols * 3, 4))
    
    # Unpack grayscale images and overlays
    preoperative_img, preoperative_overlay = preoperative
    postoperative_img, postoperative_overlay = postoperative

    # Display Preoperative Image with Overlay
    axs[0].imshow(preoperative_img, cmap="gray")
    axs[0].imshow(np.ma.masked_where(preoperative_overlay == 0, preoperative_overlay), cmap="jet", alpha=0.5, vmin=0.9, vmax=1.0)
    axs[0].axis("off")
    axs[0].set_title("Preoperative State")

    # Display Postoperative Image with Overlay
    axs[1].imshow(postoperative_img, cmap="gray")
    axs[1].imshow(np.ma.masked_where(postoperative_overlay == 0, postoperative_overlay), cmap="jet", alpha=0.5)
    axs[1].axis("off")
    axs[1].set_title("Intraoperative State")

    # Display Change Maps and Overlays
    for i, (change_map, title, score, seg_mask) in enumerate(args):
        idx = 2 + i  # Shift index after preoperative & postoperative images
        if "RiA" in title:
            change_map_mask = np.ma.masked_where(seg_mask == 0, seg_mask)
            vminx = 0.1
            alpha = 1
        else:
            change_map_norm = change_map / np.max(change_map) if np.max(change_map) > 0 else change_map
            change_map_mask = np.ma.masked_where(change_map_norm == 0, change_map_norm)
            vminx = 0
            alpha = 1

        # Change Map Visualization (Jet Colormap)
        axs[idx].imshow(change_map, cmap="jet", alpha=1)
        axs[idx].axis("off")
        axs[idx].set_title(title + f"\n{score})")
        
        # Overlay Change Map on Postoperative Image
        idx_overlay = 2 + num_maps + i  # Position after all change maps
        axs[idx_overlay].imshow(postoperative_img, cmap="gray")
        axs[idx_overlay].imshow(change_map_mask, cmap="jet", alpha=alpha, vmin=vminx, vmax=1)  # Overlay with transparency
        axs[idx_overlay].axis("off")
        axs[idx_overlay].set_title(f"{title}\n(superimposed)")
    
    # Display Ground Truth (if available and show_gt=True)
    if show_gt and ground_truth is not None:
        gt_img, gt_overlay = ground_truth
        gt_idx = -2  # Second to last column
        axs[gt_idx].imshow(gt_overlay, cmap="jet")
        axs[gt_idx].axis("off")
        axs[gt_idx].set_title("Ground Truth Mask $\Delta T$")

    # Display Final Ground Truth Overlay (if available)
    if ground_truth is not None:
        gt_img, gt_overlay = ground_truth
        axs[-1].imshow(gt_img, cmap="gray")
        axs[-1].imshow(np.ma.masked_where(gt_overlay == 0, gt_overlay), cmap="jet", alpha=0.5, vmin=0.7)
        axs[-1].axis("off")
        axs[-1].set_title("Ground Truth")

    # Adjust layout and save
    plt.tight_layout()
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

def custom_np_norm(arr, min_val=0.0, max_val=1.0):
    """Normalize a numpy array to a given range [min_val, max_val]."""
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)  # Normalize to [0,1]
    return arr * (max_val - min_val) + min_val  # Scale to [min_val, max_val]
# def multiplicative_sharpening_and_filter(distance_map: np.ndarray, base_image: np.ndarray, alpha=1.0, beta=1, threshold=0.8, max_intensity=0.6, overlay_weight=0.5):
#     """Apply sharpening while ensuring the base image remains grayscale and distance map is visualized using JET colormap."""
#     assert distance_map.max() <= 1.0, f"max: {distance_map.max()}"
#     assert distance_map.min() >= 0.0, f"min: {distance_map.min()}"
#     assert base_image.max() <= 1.0, f"max: {base_image.max()}"
#     assert base_image.min() >= 0.0, f"min: {base_image.min()}"

#     # Normalize the base image to be clearly visible in grayscale
#     base_image = custom_np_norm(base_image, min_val=0.2, max_val=0.8)

#     # Thresholding the distance map
#     distance_map = (distance_map > threshold).astype(np.float32) * distance_map  # Keep only values above threshold

#     # Extract high-frequency details from the base image
#     blurred_image = cv2.GaussianBlur(base_image, (5, 5), 0)
#     high_freq_details = base_image - blurred_image

#     # Sharpening based purely on the distance map's strength
#     sharpened_map = distance_map * (1 + alpha * high_freq_details)

#     # Ensure low base image regions still retain enhancement
#     enhanced_map = sharpened_map + beta * (1 - distance_map) * base_image

#     # Normalize enhanced map
#     enhanced_map = custom_np_norm(enhanced_map, min_val=0.0, max_val=max_intensity)

#     # Convert base image to 3-channel grayscale for visualization
#     base_gray = cv2.cvtColor((base_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#     # Convert enhanced map to JET colormap
#     distance_colormap = cv2.applyColorMap((enhanced_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

#     # Blend grayscale base image with the distance colormap
#     final_visualization = cv2.addWeighted(base_gray, (1 - overlay_weight), distance_colormap, overlay_weight, 0)

#     return final_visualization
def multiplicative_sharpening_and_filter(distance_map: np.ndarray, base_image: np.ndarray, alpha=2.0, beta=1, threshold=0.55):
    """Process the distance map with sharpening, without blending with the base image."""
    
    assert distance_map.max() <= 1.0, f"max: {distance_map.max()}"
    assert distance_map.min() >= 0.0, f"min: {distance_map.min()}"
    assert base_image.max() <= 1.0, f"max: {base_image.max()}"
    assert base_image.min() >= 0.0, f"min: {base_image.min()}"

    # Threshold and normalize the distance map
    distance_map = (distance_map > threshold).astype(np.float32) * distance_map

    # Extract high-frequency details from the base image
    blurred_image = cv2.GaussianBlur(base_image, (5, 5), 0)
    high_freq_details = base_image - blurred_image

    # Apply sharpening based purely on the distance map's strength
    sharpened_map = distance_map * (1 + alpha * high_freq_details)

    # Normalize the final enhanced distance map
    norm_enhanced_map = normalize_np_array(sharpened_map)

    return sharpened_map  



# def multiplicative_sharpening(distance_map: np.ndarray, base_image: np.ndarray, alpha=4.0, beta=0.5):
#     """Apply multiplicative sharpening by injecting high-frequency details from the base image."""
#     assert distance_map.max() <= 1.0, f"max: {distance_map.max()}"
#     assert distance_map.min() >= 0.0, f"min: {distance_map.min()}"
#     assert base_image.max() <= 1.0, f"max: {base_image.max()}"
#     assert base_image.min() >= 0.0, f"min: {base_image.min()}"

#     # Blur the base image and calculate high-frequency details
#     blurred_image = cv2.GaussianBlur(base_image, (5, 5), 0)
#     high_freq_details = base_image - blurred_image

#     # Compute sharpened map for high-value regions
#     sharpened_map = distance_map * (1 + alpha * high_freq_details)
    
#     # Add contribution from the base image in low-value regions
#     enhanced_map = sharpened_map + beta * base_image * (1 - distance_map)


#     # Normalize the final result to keep it within [0, 1]
#     enhanced_map = normalize_np_array(enhanced_map)
    
#     return enhanced_map

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

def create_histogram_f1(f1_score, save_dir, extra_title=""):
    """
    Create a histogram of F1 scores for a given dataset.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(f1_score, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("F1 Score")
    plt.ylabel("Frequency")
    plt.title("Histogram of F1 Score distribution")
    plt.savefig(os.path.join(save_dir, f'f1_hist.png'))
    plt.close()  # Close the plot to free up memory