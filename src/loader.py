import torchio as tio
import os
import nibabel as nib
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from nilearn.image import resample_to_img
from transformations import ShiftImage
from numpy import ndarray
from typing import Tuple
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
import random
from scipy.ndimage import shift, zoom
from matplotlib import pyplot as plt
def get_baseline_np(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    diff = np.abs(pre - post)
    return diff

def balance_dataset(subject_images, label_key='label'):
    """
    Balances the dataset by undersampling the majority class to match the size of the minority class.
    
    Parameters:
    subject_images (list): List of dictionaries containing images and labels.
    label_key (str): Key used to access the label in the dictionaries.
    
    Returns:
    list: Balanced list of dictionaries.
    """

    similar_pairs = [x for x in subject_images if x[label_key] == 1]
    dissimilar_pairs = [x for x in subject_images if x[label_key] == 0]
    print(f"dissimalar pairs: {len(dissimilar_pairs)}", f" ||  similar pairs: {len(similar_pairs)}")
    num_similar_pairs = len(similar_pairs)
    num_dissimilar_pairs = len(dissimilar_pairs)

    if num_similar_pairs > num_dissimilar_pairs:
        similar_pairs = random.sample(similar_pairs, num_dissimilar_pairs)
    else:
        dissimilar_pairs = random.sample(dissimilar_pairs, num_similar_pairs)
    balanced_subject_images = similar_pairs + dissimilar_pairs
    random.shuffle(balanced_subject_images)  # Shuffle to mix the pairs
    return balanced_subject_images

def stratified_kfold_split(dataset, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    splits = list(skf.split(range(len(dataset)), dataset.labels))
    return splits

def normalize_np_array(array: ndarray) -> ndarray:
    if np.max(array) == np.min(array):
        return array
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def normalize_nifti(nifti_image: nib.Nifti1Image) -> ndarray:
    return (nifti_image.get_fdata() - np.min(nifti_image.get_fdata())) / (np.max(
        nifti_image.get_fdata()) - np.min(nifti_image.get_fdata()))

def pad_slice(slice_2d: ndarray, output_size=(256, 256)) -> ndarray:
    """
    Pad a 2D slice to the desired output size with zeros.
    """
    pad_height = (output_size[0] - slice_2d.shape[0])
    pad_width = (output_size[1] - slice_2d.shape[1])
    
    padded_slice = np.pad(slice_2d, 
                          ((0, pad_height), (0, pad_width)), 
                          mode='edge')
    return padded_slice

def slice_has_high_info(slice_2d: np.ndarray, value_minimum=0.15, percentage_minimum=0.12):
    ## checks if the slice has high information by a certain value threshold and percentage of cells
    total_cells = slice_2d.size
    num_high_info_cells = np.count_nonzero(slice_2d >= value_minimum)
    percentage_high_info = num_high_info_cells / total_cells
    return percentage_high_info > percentage_minimum

def convert_3d_into_2d(nifti_image: ndarray, skip: int =1) -> list[Tuple[ndarray, Tuple[int, int, int]]]:
    slices = []
   
    # (axial)
    ## TODO: Use all slices for now only using every 4th slices
    for i in range(nifti_image.shape[0]):
        if i % skip == 0:
            slices.append((nifti_image[i, :, :], (i, -1 , -1)))
    #  (coronal)
    for i in range(nifti_image.shape[1]):
        if i % skip == 0:
            slices.append((nifti_image[:, i, :], (-1, i, -1)))  
    # (sagittal)
    for i in range(nifti_image.shape[2]):
        if i % skip == 0:
            slices.append((nifti_image[:, :, i], (-1, -1, i)))
    return slices

def array_has_significant_values(slice_2d: ndarray, threshold=0.15):
    ## checks if the slice has tumor cells by a certain value threshold
    return np.any(slice_2d >= threshold)

def filter_array_on_threshold(slice_2d: ndarray, threshold=0.15) -> ndarray:
    """
    Keeps only the values in the slice that are above the threshold.

    Args:
        slice_2d (ndarray): Input 2D slice array.
        threshold (float): Value threshold to consider a cell has a significant value 
        (changes or tumor for example).

    Returns:
        ndarray: The modified array with values below the threshold set to zero.
    """
    # Set values below the threshold to zero in place
    slice_2d[slice_2d < threshold] = 0
    
    return slice_2d

def convert_tuple_to_string(index):
    if index[0] != -1:
        return "axial_" + str(index[0])
    elif index[1] != -1:
        return "coronal_" + str(index[1])
    elif index[2] != -1:
        return "sagittal_" + str(index[2])
def shift_image_numpy(image: np.ndarray, shift_amount: tuple, fill_value=0, mode='nearest') -> np.ndarray:
    """
    Shift an image using NumPy and fill the edges with the specified fill_value.
    
    Args:
        image (np.ndarray): Input 2D or 3D image array.
        shift_amount (tuple): Tuple indicating the amount to shift along each axis.
        fill_value (int, float, optional): Value to use for edge filling. Defaults to 0.
    
    Returns:
        np.ndarray: Shifted image with edges filled.
    """

    shifted_image = shift(image, shift=shift_amount, cval=fill_value, mode=mode)
    return shifted_image

def downsize_if_needed_array(image_array: ndarray, target = 256) -> ndarray:
    # Get the original size
    original_height, original_width = image_array.shape

    if original_height <= target and original_width <= target:
        return image_array

    if original_height > original_width:
        # Set height to fixed_dim
        new_height = target
        resize_factor = new_height / original_height
        new_width = int(original_width * resize_factor)
    else:
        # Set width to fixed_dim
        new_width = target
        resize_factor = new_width / original_width
        new_height = int(original_height * resize_factor)

    # Resize the image using zoom with cubic interpolation
    resized_image = zoom(image_array, (resize_factor, resize_factor), order=0)

    return resized_image
def reorient_to_standard(image):
    # Get the current orientation as axis codes
    current_orientation = nib.aff2axcodes(image.affine)
    # Define the desired orientation (RAS)
    desired_orientation = ('R', 'A', 'S')

    # Convert axis codes to orientation arrays
    current_ornt = nib.orientations.axcodes2ornt(current_orientation)
    desired_ornt = nib.orientations.axcodes2ornt(desired_orientation)

    # Get the orientation transform
    ornt_transform = nib.orientations.ornt_transform(current_ornt, desired_ornt)

    # Apply the orientation transform
    reoriented_data = nib.orientations.apply_orientation(image.get_fdata(), ornt_transform)
    reoriented_affine = nib.orientations.inv_ornt_aff(ornt_transform, image.shape)
    reoriented_affine = np.dot(image.affine, reoriented_affine)

    # Create a new NIfTI image with the reoriented data and affine
    reoriented_image = nib.Nifti1Image(reoriented_data, reoriented_affine)
    return reoriented_image

def find_matching_file(directory, image_id):
    for root, _, files in os.walk(directory):
        for file in files:
            if image_id in file:
                return os.path.join(root, file)
    return None
def is_preop_the_target_shape(preop_shape: Tuple[int, int, int], postop_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    # Calculate the aspect ratios for each dimension
    preop_aspect_ratio = np.std([preop_shape[0] / preop_shape[1], preop_shape[1] / preop_shape[2], preop_shape[2] / preop_shape[0]])
    postop_aspect_ratio = np.std([postop_shape[0] / postop_shape[1], postop_shape[1] / postop_shape[2], postop_shape[2] / postop_shape[0]])

    # Choose the shape with the aspect ratio closest to 1
    if preop_aspect_ratio < postop_aspect_ratio:
        return True
    else:
        return False

def save_before_comparison_with_tumor(pre_slice: np.ndarray, post_slice: np.ndarray, mask_slice: np.ndarray, pat_id: str, index: Tuple[int, int, int], label: int, save_dir: str, color='hot') -> str:
    """Save the comparison image with tumor overlay, pre slice, and post slice."""
    brain_axis = convert_tuple_to_string(index)
    filename = f"{pat_id}_slice_{brain_axis}_{label}.png"
    save_path = os.path.join(save_dir, 'overview', filename)

    # Create the tumor overlay on the pre slice
    tumor_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    tumor_overlay_normalized = tumor_overlay / np.max(tumor_overlay) if np.max(tumor_overlay) > 0 else tumor_overlay

    # Plot the images
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # First image: pre slice with tumor overlay
    ax[0].imshow(pre_slice, cmap='gray')
    ax[0].imshow(tumor_overlay_normalized, cmap=color, alpha=1)
    ax[0].axis('off')
    ax[0].set_title('Pre Slice with Tumor Overlay')

    # Second image: regular pre slice
    ax[1].imshow(pre_slice, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Pre Slice')

    # Third image: post slice
    ax[2].imshow(post_slice, cmap='gray')
    ax[2].axis('off')
    ax[2].set_title('Post Slice')

    # Save the figure
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    return save_path

def return_relative_tumor_overlay(mask_slice: np.ndarray) -> str:
    """save tumor mask relative to the pre image"""

    # Create the tumor overlay on the pre slice
    tumor_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    tumor_overlay_normalized = tumor_overlay / np.max(tumor_overlay) if np.max(tumor_overlay) > 0 else tumor_overlay
    return tumor_overlay_normalized

def get_index_tuple(index: int, orientation: str) -> Tuple[int, int, int]:
    """Convert index to a tuple based on orientation."""
    if orientation == 'axial':
        return (index, -1, -1)
    elif orientation == 'coronal':
        return (-1, index, -1)
    elif orientation == 'sagittal':
        return (-1, -1, index)
    return (-1, -1, -1)

class remindDataset(Dataset):
    def __init__(self, preop_dir: str, image_ids: list, save_dir: str, skip:int=1, tumor_sensitivity = 0.10, load_slices=False, transform=None):
        self.preop_dir = preop_dir
        self.transform = transform
        self.image_ids = image_ids
        self.save_dir = save_dir  # Directory to save 2D slices
        self.skip = skip
        self.tumor_sensitivity = tumor_sensitivity
        self.data = []
        self.load_slices = load_slices
        self.preop_dir = preop_dir
        # https://www.nature.com/articles/s41597-024-03295-z
        print("Starting remind dataset")
        
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure the save directory exists
        os.makedirs(os.path.join(self.save_dir, 'overview'), exist_ok=True)  # Ensure the overview directory exists
        self._find_nifti_patients_and_get_slices()

    def _find_nifti_patients_and_get_slices(self):
        for root_path, dirs, files in os.walk(self.preop_dir):
            for filename in files:
                for image_id in self.image_ids:
                    if filename.startswith(image_id):
                        try:
                            pat_id = root_path.split("/")[-3]
                            if "Intraop" in root_path or "Unused" in root_path:
                                continue
                            print(f"Processing {pat_id}, loading {filename}")
                            if self.load_slices:
                                print(f"Loading existing slices for {pat_id} instead of processing...")
                                self._load_existing_slices(pat_id)
                                continue
                            preop_nifti = nib.load(os.path.join(root_path, filename))
                            post_op_path = root_path.replace("Preop", "Intraop")
                            matching_filename = find_matching_file(post_op_path, image_id)
                            
                            if matching_filename:
                                postop_nifti = nib.load(matching_filename)
                                print(f"Found matching Intraop file: {matching_filename}")
                            else:
                                print(f"No matching Intraop file found for image_id: {image_id}")
                                continue
                            tumor = nib.load(os.path.join(root_path.replace("T1_converted", "tumor_converted"), 
                                                          "1.nii.gz"))
                            
                            assert preop_nifti != None and postop_nifti != None and tumor != None
                            if is_preop_the_target_shape(preop_nifti.shape, postop_nifti.shape):
                                print(f"Resampling to preop shape {preop_nifti.shape}, compared to postop shape {postop_nifti.shape}")
                                if nib.aff2axcodes(preop_nifti.affine) != ('R', 'A', 'S'):
                                    preop_nifti = reorient_to_standard(preop_nifti)
                                postop_nifti = resample_to_img(source_img=postop_nifti, target_img=preop_nifti, interpolation='nearest')
                                tumor_resampled = resample_to_img(source_img=tumor, target_img=preop_nifti, interpolation='nearest')
                            else:
                                if nib.aff2axcodes(postop_nifti.affine) != ('R', 'A', 'S'):
                                    postop_nifti = reorient_to_standard(postop_nifti)
                                preop_nifti = resample_to_img(source_img=preop_nifti, target_img=postop_nifti, interpolation='nearest')
                                tumor_resampled = resample_to_img(source_img=tumor, target_img=postop_nifti, interpolation='nearest')

                            assert nib.aff2axcodes(preop_nifti.affine) == nib.aff2axcodes(postop_nifti.affine) == nib.aff2axcodes(tumor_resampled.affine) == ('R', 'A', 'S'), "ArithmeticError: Affine mismatch"
                            tumor_norm = normalize_nifti(tumor_resampled)
                            preop_nifti_norm = normalize_nifti(preop_nifti)
                            postop_nifti_norm = normalize_nifti(postop_nifti)
                            assert preop_nifti_norm.max() <= 1.0, f"max: {preop_nifti_norm.max()}"
                            assert postop_nifti_norm.min() >= 0.0, f"min: {postop_nifti_norm.min()}"
                            # Convert 3D images to 2D slices
                            images_and_indices_pre = convert_3d_into_2d(preop_nifti_norm, skip=self.skip)
                            images_and_indices_post = convert_3d_into_2d(postop_nifti_norm, skip=self.skip)
                            masks_and_indices = convert_3d_into_2d(tumor_norm, skip=self.skip)
                            self._process_pat_slices(pat_id, 
                                                     images_and_indices_pre, 
                                                     images_and_indices_post, 
                                                     masks_and_indices)
                        except FileNotFoundError as e:
                            print(f"File not found: {e}")
                        except AssertionError as e:
                            print(f"AssertionError: {e}")
                        except Exception as e:
                            print(f"Uncaught error: {e}")

    def _load_existing_slices(self, pat_id: str):
        """Load existing slices for the given pat_id."""
        orientations = ['axial', 'coronal', 'sagittal']
        for orientation in orientations:
            pre_slices = [f for f in os.listdir(self.save_dir) if f"{pat_id}_slice_{orientation}" in f and "_pre_" in f]
            # print(f"{pat_id}_slice_{orientation}_pre")
            pre_slices = pre_slices[::self.skip]
            for pre_slice in pre_slices:
                pre_path = os.path.join(self.save_dir, pre_slice)
                post_slice = pre_slice.replace('_pre_', '_post_')
                post_path = os.path.join(self.save_dir, post_slice)
                if not os.path.exists(post_path):
                    continue
                tumor_path = ""
                if "ReMIND" in pat_id:
                    tumor_slice = pre_slice.replace('_pre_', '_tumor_')
                    tumor_path = os.path.join(self.save_dir, tumor_slice)
                    if not os.path.exists(tumor_path):
                        tumor_path = ""
                label = int(pre_slice.split('_')[-1].split('.')[0])
                index_pre = int(pre_slice.split('_')[3])
                index_post = int(post_slice.split('_')[3])
                assert index_pre == index_post, f"Indices do not match: {index_pre}, {index_post}"
                index_tuple = get_index_tuple(index_pre, orientation)

                self.data.append({
                    "pre_path": pre_path,
                    "post_path": post_path,
                    "tumor_path": tumor_path,
                    "label": label,
                    "pat_id": pat_id,
                    "index_pre": index_tuple,
                    "index_post": index_tuple,
                })
            print(f"loaded {len(pre_slices)} existing slices for {pat_id}")
    def _process_pat_slices(self, pat_id: str, 
                            images_pre: list[Tuple[ndarray, Tuple[int, int, int]]], 
                            images_post: list[Tuple[ndarray, Tuple[int, int, int]]], 
                            mask_slices: list[Tuple[ndarray, Tuple[int, int, int]]]):
        """Process patient (PAT) slices and save them."""
        for (pre_slice_and_index, post_slice_and_index, mask_slice_and_index) in  zip(images_pre, images_post, mask_slices):
            pre_slice_padded = pad_slice(downsize_if_needed_array(pre_slice_and_index[0]))
            post_slice_padded = pad_slice(downsize_if_needed_array(post_slice_and_index[0]))
            mask_slice_and_index = pad_slice(downsize_if_needed_array(mask_slice_and_index[0])), mask_slice_and_index[1]
            pre_index = pre_slice_and_index[1]
            post_index = post_slice_and_index[1]
            mask_index = mask_slice_and_index[1]
            assert pre_index == post_index == mask_index, f"Indices\
                tuples do not match: {pre_index}, {post_index}, {mask_index}"
                
            assert pre_slice_padded.shape == post_slice_padded.shape  == (256, 256), f"Shapes do not match: {pre_slice_padded.shape}, {post_slice_padded.shape}"
            label = 0 if array_has_significant_values(mask_slice_and_index[0], threshold=self.tumor_sensitivity) else 1
            #NOTE: skipping control pairs
            if label == 1:
                if slice_has_high_info(pre_slice_padded, 0.15, 0.15) and slice_has_high_info(post_slice_padded, 0.15, 0.15):
                    pre_path = self._save_slice(pre_slice_padded, pat_id, pre_index, 'pre', label)
                    post_path = self._save_slice(post_slice_padded, pat_id, post_index, 'post', label)
                    tumor_path = self._save_slice(mask_slice_and_index[0], pat_id, mask_index, 'tumor', label)
                    pre_post_tumor_vis = save_before_comparison_with_tumor(pre_slice_padded, post_slice_padded, mask_slice_and_index[0], pat_id, pre_index, label, self.save_dir)
                
                    self.data.append({"pre_path": pre_path, "post_path": post_path, 
                                    "tumor_path": tumor_path, "label": label, "pat_id": pat_id,
                                    "index_pre": pre_index, "index_post": post_index})
            else:
            ##NOTE: we are not using remind for control pairs so we don't need to check for high info
            ## because sometimes this filters too strongly since the images are heavily padded so we loosen the percentage minimum
                if slice_has_high_info(pre_slice_padded, value_minimum=0.15, percentage_minimum=0.001) and slice_has_high_info(post_slice_padded, percentage_minimum=0.15, value_minimum=0.001):
                    pre_path = self._save_slice(pre_slice_padded, pat_id, pre_index, 'pre', label)
                    post_path = self._save_slice(post_slice_padded, pat_id, post_index, 'post', label)
                    tumor_path = self._save_slice(mask_slice_and_index[0], pat_id, mask_index, 'tumor', label)
                    pre_post_tumor_vis = save_before_comparison_with_tumor(pre_slice_padded, post_slice_padded, mask_slice_and_index[0], pat_id, pre_index, label, self.save_dir, 'jet')
                
                    self.data.append({"pre_path": pre_path, "post_path": post_path, 
                                    "tumor_path": tumor_path, "label": label, "pat_id": pat_id,
                                    "index_pre": pre_index, "index_post": post_index})


    def _save_slice(self, slice_array: ndarray, pat_id: str, index: Tuple, slice_type: str, label: int):
        """Save the 2D slice as a numpy file and return the file path."""
        brain_axis = convert_tuple_to_string(index)
        ##TODO: remove label from this
        filename = f"{pat_id}_slice_{brain_axis}_{slice_type}_label_{label}.npz"
        save_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(save_path):
            np.savez_compressed(save_path, data=slice_array)           
        return save_path
    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        triplet = self.data[idx]
        pre_slice = np.load(triplet["pre_path"])['data']
        post_slice = np.load(triplet["post_path"])['data']
        tumor_slice = np.load(triplet["tumor_path"])['data'] if triplet["tumor_path"] else np.zeros_like(pre_slice) 
        baseline = get_baseline_np(pre_slice, post_slice)
        assert pre_slice.shape == post_slice.shape == (256, 256), f"Shapes do not match: {pre_slice.shape}, {post_slice.shape}"
        # tumor_slice = np.load(triplet["tumor_path"]) if "tumor_path" in triplet else None

        # Apply any transformations if necessary
        shift_x = torch.randint(-50, 51, (1,)).item()
        shift_y = torch.randint(-50, 51, (1,)).item()
        shift = (shift_x, shift_y)
        ## HACK: to get the preop tumor slice and postop tumor slice
        ## to later use for feature map evaluation
        if self.transform:
            for transform in self.transform.transforms:
                if isinstance(transform, ShiftImage):
                    pre_tumor = tumor_slice
                    post_slice = transform(post_slice, shift=shift)
                    tumor_slice = transform(tumor_slice, shift=shift)
                else:
                    pre_slice = transform(pre_slice)
                    post_slice = transform(post_slice)
                    tumor_slice = transform(tumor_slice)

        return {"pre": pre_slice, "post": post_slice, "label": triplet["label"], 
                "pat_id": triplet["pat_id"], "index_pre": triplet["index_pre"], 
                "index_post": triplet["index_post"],
                "baseline": baseline, "post_tumor": tumor_slice, "pre_tumor": pre_tumor,
                "pre_path": triplet["pre_path"], "tumor_path": triplet["tumor_path"]}
        
class aertsDataset(Dataset):
    def __init__(self, proc_preop: str, raw_tumor_dir: str, image_ids: list, save_dir: str, skip:int=1, tumor_sensitivity = 0.10, load_slices = False, transform=None):
        self.root = proc_preop
        self.raw_tumor_dir = raw_tumor_dir
        self.transform = transform
        self.image_ids = image_ids
        self.save_dir = save_dir  # Directory to save 2D slices
        self.skip = skip
        self.tumor_sensitivity = tumor_sensitivity
        self.data = []
        self.load_slices = load_slices
        # https://www.nature.com/articles/s41597-022-01806-4

        os.makedirs(self.save_dir, exist_ok=True)  # Ensure the save directory exists
        os.makedirs(os.path.join(self.save_dir, 'overview'), exist_ok=True)  # Ensure the overview directory exists
        self._find_nifti_patients_and_save_slices()


    def _load_existing_slices(self, pat_id: str):
        """Load existing slices for the given pat_id."""
        orientations = ['axial', 'coronal', 'sagittal']
        for orientation in orientations:
            pre_slices = [f for f in os.listdir(self.save_dir) if f"{pat_id}_slice_{orientation}" in f and "_pre_" in f]
            # print(f"{pat_id}_slice_{orientation}_pre")
            pre_slices = pre_slices[::self.skip]

            for pre_slice in pre_slices:
                pre_path = os.path.join(self.save_dir, pre_slice)
                post_slice = pre_slice.replace('_pre_', '_post_')
                post_path = os.path.join(self.save_dir, post_slice)
                if not os.path.exists(post_path):
                    continue
                tumor_path = ""
                if "PAT" in pat_id:
                    tumor_slice = pre_slice.replace('_pre_', '_tumor_')
                    tumor_path = os.path.join(self.save_dir, tumor_slice)
                    if not os.path.exists(tumor_path):
                        tumor_path = ""
                label = int(pre_slice.split('_')[-1].split('.')[0])
                index_pre = int(pre_slice.split('_')[3])
                index_post = int(post_slice.split('_')[3])
                assert index_pre == index_post, f"Indices do not match: {index_pre}, {index_post}"
                index_tuple = get_index_tuple(index_pre, orientation)

                self.data.append({
                    "pre_path": pre_path,
                    "post_path": post_path,
                    "tumor_path": tumor_path,
                    "label": label,
                    "pat_id": pat_id,
                    "index_pre": index_tuple,
                    "index_post": index_tuple,
                })
            print(f"loaded {len(pre_slices)} existing slices for {pat_id}")
                
    def _find_nifti_patients_and_save_slices(self):
        for root, dirs, files in os.walk(self.root):
            for filename in files:
                for image_id in self.image_ids:
                    if filename.endswith(image_id):
                        try:
                            pat_id = root.split("/")[-1]
                            if pat_id == "sub-PAT24":
                                print("skippging pat24 since it has a bad tumor")
                                continue
                            print(f"Processing {pat_id}")
                            if self.load_slices:
                                print(f"Loading existing slices for {pat_id} instead of processing...")
                                self._load_existing_slices(pat_id)
                                continue
                            preop_nifti = nib.load(os.path.join(root, filename))

                            postop_nifti = nib.load(os.path.join(root.replace("preop", "postop"), 
                                                                filename.replace("preop", "postop")))

                            if "PAT" in pat_id:
                                try:
                                    tumor = nib.load(os.path.join(f"{self.raw_tumor_dir}/{pat_id}/anat/{pat_id}_space_T1_label-tumor.nii"))
                                    tumor_resampled = resample_to_img(tumor, preop_nifti, interpolation='nearest')
                                    tumor_norm = normalize_nifti(tumor_resampled)
                                    assert tumor_norm.max() <= 1.0, f"max: {tumor_norm.max()}"

                                except FileNotFoundError:
                                    print(f"Tumor not found for {pat_id}")
                                    tumor_norm = None

                            # Normalize and resample preop and postop images
                            preop_nifti_norm = normalize_nifti(preop_nifti)
                            postop_nifti_norm = normalize_nifti(postop_nifti)
                            assert preop_nifti_norm.max() <= 1.0, f"max: {preop_nifti_norm.max()}"
                            assert postop_nifti_norm.min() >= 0.0, f"min: {postop_nifti_norm.min()}"

                            # Convert 3D images to 2D slices
                            
                            images_pre = convert_3d_into_2d(preop_nifti_norm, skip=self.skip)
                            images_post = convert_3d_into_2d(postop_nifti_norm, skip=self.skip)

                            if "-CON" in pat_id:
                                self._process_con_slices(pat_id, images_pre, images_post)
                            elif "-PAT" in pat_id and tumor_norm is not None:
                                mask_slices = convert_3d_into_2d(tumor_norm, skip=self.skip)
                                self._process_pat_slices(pat_id, images_pre, images_post, mask_slices)
                        except FileNotFoundError as e:
                            print(f"File not found: {e}")
                        except Exception as e:
                            print(f"Uncaught error: {e}")

    def _process_con_slices(self, pat_id: str, 
                            images_pre: list[Tuple[ndarray, Tuple[int, int, int]]], 
                            images_post: list[Tuple[ndarray, Tuple[int, int, int]]]):
        """Process control patient (CON) slices and save them."""
        for (pre_slice_and_index, post_slice_and_index) in  zip(images_pre, images_post):
            pre_slice_padded = pad_slice(downsize_if_needed_array(pre_slice_and_index[0]))
            post_slice_padded = pad_slice(downsize_if_needed_array(post_slice_and_index[0]))

            pre_slice_index= pre_slice_and_index[1]
            post_slice_index= post_slice_and_index[1]
            
            assert pre_slice_index == post_slice_index, f"Indices tuples do not match: {pre_slice_index}, {post_slice_index}"
            assert pre_slice_padded.shape == post_slice_padded.shape == (256, 256), f"Shapes do not match: {pre_slice_padded.shape}, {post_slice_padded.shape}"
            if slice_has_high_info(pre_slice_padded) and slice_has_high_info(post_slice_padded):
                pre_path = self._save_slice(pre_slice_padded, pat_id, pre_slice_index, 'pre', 1)
                post_path = self._save_slice(post_slice_padded, pat_id, post_slice_index, 'post', 1)
                self.data.append({"pre_path": pre_path, "post_path": post_path, "label": 1, "pat_id": pat_id,
                                  "index_pre": pre_slice_index, "index_post": post_slice_index, "tumor_path": ""})

    def _process_pat_slices(self, pat_id: str, 
                            images_pre: list[Tuple[ndarray, Tuple[int, int, int]]], 
                            images_post: list[Tuple[ndarray, Tuple[int, int, int]]], 
                            mask_slices: list[Tuple[ndarray, Tuple[int, int, int]]]):
        """Process patient (PAT) slices and save them."""
        for (pre_slice_and_index, post_slice_and_index, mask_slice_and_index) in  zip(images_pre, images_post, mask_slices):
            pre_slice_padded = pad_slice(downsize_if_needed_array(pre_slice_and_index[0]))
            post_slice_padded = pad_slice(downsize_if_needed_array(post_slice_and_index[0]))
            mask_slice_and_index = pad_slice(downsize_if_needed_array(mask_slice_and_index[0])), mask_slice_and_index[1]
            
            pre_slice_index = pre_slice_and_index[1]
            post_slice_index = pre_slice_and_index[1]
            tumor_slice_index = pre_slice_and_index[1]
            assert pre_slice_index == post_slice_index == tumor_slice_index, f"Indices\
                tuples do not match: {pre_slice_index}, {post_slice_index}, {tumor_slice_index}"
                
            assert pre_slice_padded.shape == post_slice_padded.shape  == (256, 256), f"Shapes do not match: {pre_slice_padded.shape}, {post_slice_padded.shape}"
            label = 0 if array_has_significant_values(mask_slice_and_index[0], threshold=self.tumor_sensitivity) else 1
            if label == 1:
                percentage_min = 0.12
            else:
                percentage_min = 0.005
            #NOTE: since we are NOT using remind for control pairs (label 1) we don't need to check for high info
            # because sometimes this filters too strongly since the images are heavily padded so we loosen the percentage minimum
            if slice_has_high_info(pre_slice_padded, value_minimum=0.15, percentage_minimum=percentage_min) and slice_has_high_info(post_slice_padded, value_minimum=0.15, percentage_minimum=percentage_min):
                pre_path = self._save_slice(pre_slice_padded, pat_id, pre_slice_index, 'pre', label)
                post_path = self._save_slice(post_slice_padded, pat_id, post_slice_index, 'post', label)
                tumor_path = self._save_slice(mask_slice_and_index[0], pat_id, tumor_slice_index, 'tumor', label)
                pre_post_tumor_vis = save_before_comparison_with_tumor(pre_slice_padded, post_slice_padded, mask_slice_and_index[0], pat_id, pre_slice_index, label, self.save_dir)
                self.data.append({"pre_path": pre_path, "post_path": post_path, 
                                  "tumor_path": tumor_path, "label": label, "pat_id": pat_id,
                                  "index_pre": pre_slice_index, "index_post": post_slice_index})

    def _save_slice(self, slice_array: ndarray, pat_id: str, index: Tuple, slice_type: str, label: int):
        """Save the 2D slice as a numpy file and return the file path."""
        brain_axis = convert_tuple_to_string(index)
        filename = f"{pat_id}_slice_{brain_axis}_{slice_type}_label_{label}.npz"
        save_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(save_path):
            #plt.imsave(save_path + '.png', slice_array, cmap='gray')
            np.savez_compressed(save_path, data=slice_array)
        return save_path
    
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        triplet = self.data[idx]
        pre_slice = np.load(triplet["pre_path"])['data']
        post_slice = np.load(triplet["post_path"])['data']
        tumor_slice = np.load(triplet["tumor_path"])['data'] if triplet["tumor_path"] else np.zeros_like(pre_slice) 
        baseline = get_baseline_np(pre_slice, post_slice)
        assert pre_slice.shape == post_slice.shape == (256, 256), f"Shapes do not match: {pre_slice.shape}, {post_slice.shape}"
        # tumor_slice = np.load(triplet["tumor_path"]) if "tumor_path" in triplet else None

        # Apply any transformations if necessary
        shift_x = torch.randint(-50, 101, (1,)).item()
        shift_y = torch.randint(-50, 101, (1,)).item()
        shift = (shift_x, shift_y)
        ## HACK: to get the preop tumor slice and postop tumor slice
        ## to later use for feature map evaluation
        if self.transform:
            for transform in self.transform.transforms:
                if isinstance(transform, ShiftImage):
                    pre_tumor = tumor_slice
                    post_slice = transform(post_slice, shift=shift)
                    tumor_slice = transform(tumor_slice, shift=shift)
                else:
                    pre_slice = transform(pre_slice)
                    post_slice = transform(post_slice)
                    tumor_slice = transform(tumor_slice)
        
        return {"pre": pre_slice, "post": post_slice, "label": triplet["label"], 
                "pat_id": triplet["pat_id"], "index_pre": triplet["index_pre"], 
                "index_post": triplet["index_post"],
                "baseline": baseline, "post_tumor": tumor_slice, "pre_tumor": pre_tumor,
                "pre_path": triplet["pre_path"], "tumor_path": triplet["tumor_path"]}
        
        

                

class control_pairs(Dataset):
    """
    Image dataset for each subject in the dataset
    creating only control pairs, of the same image for sanity test

    Works by matching the image just by itself
    """
    def __init__(self, proc_preop: str, image_ids: list, transform=None, skip:int=1):
        self.root = proc_preop
        self.transform = transform
        self.data = []
        self.labels = [] # used for kfold later on
        self.image_ids = image_ids
        for root, dirs, files in os.walk(self.root):
            for filename in files:
                for image_id in self.image_ids:
                    if filename.endswith(image_id):
                        try:
                            pat_id = root.split("/")[-1]
                            print(f"Processing {pat_id}")
                            preop_nifti = nib.load(os.path.join(root, filename))
                            # resample the postop nifti to the preop nifti
                            preop_nifti_norm = normalize_nifti(preop_nifti)

                            assert preop_nifti_norm.max() <= 1.0, f"max: {preop_nifti_norm.max()}"
                            assert preop_nifti_norm.min() >= 0.0, f"min: {preop_nifti_norm.min()}"

                            
                            images_pre = convert_3d_into_2d(preop_nifti_norm, skip=skip)

                            # Create triplets with label 1 (similar slices)
                            images_pre_pad = [(pad_slice(image[0]), image[1], 1) for image in images_pre]
    
                            
                            # Create triplets (pre_slice, post_slice, label, tumor=None)
                            triplets_con = [{"pre": pre, "post": post, "label": label, "tumor": np.zeros_like(pre), 
                                                "pat_id": pat_id, "index_pre": index_pre, "index_post": index_post} 
                                            for (pre, index_pre, label), 
                                            (post, index_post, _) in 
                                            zip(images_pre_pad, images_pre_pad) if 
                                            slice_has_high_info(pre) and slice_has_high_info(post)]
                            self.data.extend(triplets_con)
                            self.labels.extend([label for (_, label, _) in images_pre_pad])
                        
                        except FileNotFoundError as e:
                            print(f"{e}, this is normal to happen for 3 subjects which have no postoperative data")
                        except Exception as e:
                            print(f"Uncaught error, {e}")
                    else:
                        pass
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):

        if self.transform:
            pass
        return self.data[idx]


def transform_subjects(subjects: list[tio.Subject]) -> tio.SubjectsDataset:
    transforms = [
    # tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.CropOrPad((164, 164, 164)),
    ]
    transform = tio.Compose(transforms)
    return tio.SubjectsDataset(subjects, transform=transform)

