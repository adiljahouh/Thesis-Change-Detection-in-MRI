import torch.nn.functional as F
import numpy as np
def various_distance(out_vec_t0, out_vec_t1,dist_flag):
    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
    if dist_flag == 'cos':
        distance = 1 - F.cosine_similarity(out_vec_t0, out_vec_t1)
    return distance

def threshold_by_zscore_std(diff_map, threshold=4):
    """Thresholds a difference map using Z-score significance."""
    mean_val = np.mean(diff_map)
    std_val = np.std(diff_map)

    # Compute Z-score
    z_scores = (diff_map - mean_val) / std_val

    # Keep only values beyond the threshold
    significant_map = np.abs(z_scores) > threshold

    return significant_map.astype(np.uint8)  # Convert to binary mask

def threshold_by_percentile(diff_map, percentile=95):
    """Thresholds a difference map using a percentile-based threshold."""
    threshold_value = np.percentile(np.abs(diff_map), percentile)  # Compute the threshold at the given percentile
    significant_map = np.abs(diff_map) > threshold_value  # Threshold the absolute differences

    return significant_map.astype(np.uint8)  # Convert to binary mask
