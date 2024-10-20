import torchio as tio
import os
import nibabel as nib
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from nilearn.image import resample_to_img
from numpy import ndarray
from typing import Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms.functional as F
import random
import torch
from scipy.ndimage import shift
import kornia.geometry.transform as kornia_transform
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

def has_tumor_cells(slice_2d: ndarray, threshold=0.15):
    ## checks if the slice has tumor cells by a certain value threshold
    return np.any(slice_2d >= threshold)

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



class ShiftImage:
    def __init__(self, max_shift_x, max_shift_y):
        self.max_shift_x = max_shift_x
        self.max_shift_y = max_shift_y

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Ensure the tensor is 3D
        shift_x = torch.randint(-self.max_shift_x, self.max_shift_x + 1, (1,)).double()
        shift_y = torch.randint(-self.max_shift_y, self.max_shift_y + 1, (1,)).double()

        # Create translation tensor
        translation = torch.tensor([[shift_x.item(), shift_y.item()]]).double()
        return kornia_transform.translate(tensor.unsqueeze(0).double(), translation, mode='bilinear', padding_mode='border', align_corners=True).squeeze(0).float()


class remindDataset(Dataset):
    def __init__(self, preop_dir: str, image_ids: list, save_dir: str, skip:int=1, tumor_sensitivity = 0.10, transform=None):
        self.preop_dir = preop_dir
        self.transform = transform
        self.image_ids = image_ids
        self.save_dir = save_dir  # Directory to save 2D slices
        self.skip = skip
        self.tumor_sensitivity = tumor_sensitivity
        self.data = []
        # https://www.nature.com/articles/s41597-024-03295-z
        
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure the save directory exists
        for root_path, dirs, files in os.walk(preop_dir):
            for filename in files:
                for image_id in self.image_ids:
                    if filename.endswith(image_id):
                        try:
                            pat_id = root_path.split("/")[-3]
                            if "Intraop" in root_path:
                                continue
                            print(f"Processing {pat_id}")
                            preop_nifti = nib.load(os.path.join(root_path, filename))
                            postop_nifti = nib.load(os.path.join(root_path.replace("Preop", "Intraop"), 
                                                                filename))
                            postop_nifti = resample_to_img(source_img=postop_nifti, target_img=preop_nifti, interpolation='nearest')
                            tumor = nib.load(os.path.join(root_path.replace("T1_converted", "tumor_converted"), "1.nii.gz"))
                            tumor_resampled = resample_to_img(source_img=tumor, target_img=preop_nifti, interpolation='nearest')
                            tumor_norm = normalize_nifti(tumor_resampled)
                            # Normalize and resample preop and postop images
                            preop_nifti_norm = normalize_nifti(preop_nifti)
                            postop_nifti_norm = normalize_nifti(postop_nifti)
                            assert preop_nifti_norm.max() <= 1.0, f"max: {preop_nifti_norm.max()}"
                            assert postop_nifti_norm.min() >= 0.0, f"min: {postop_nifti_norm.min()}"
                            print(preop_nifti_norm.shape, postop_nifti_norm.shape, tumor_norm.shape)
                            # Convert 3D images to 2D slices
                            images_pre = convert_3d_into_2d(preop_nifti_norm, skip=self.skip)
                            images_post = convert_3d_into_2d(postop_nifti_norm, skip=self.skip)
                            mask_slices = convert_3d_into_2d(tumor_norm, skip=self.skip)
                            self._process_pat_slices(pat_id, images_pre, images_post, mask_slices)
                            return
                        except FileNotFoundError as e:
                            print(f"File not found: {e}")
                        except Exception as e:
                            print(f"Uncaught error: {e}")

    def _process_pat_slices(self, pat_id, images_pre, images_post, mask_slices):
        """Process patient (PAT) slices and save them."""
        for i, (pre_slice, post_slice, mask_slice) in enumerate(zip(images_pre, images_post, mask_slices)):
            pre_slice_pad = pad_slice(pre_slice[0])
            post_slice_pad = pad_slice(post_slice[0])
            
            pre_slice_index: Tuple[int, int, int] = pre_slice[1]
            post_slice_index: Tuple[int, int, int] = post_slice[1]
            tumor_slice_index: Tuple[int, int, int] = mask_slice[1]
            assert pre_slice_index == post_slice_index == tumor_slice_index, f"Indices\
                tuples do not match: {pre_slice_index}, {post_slice_index}, {tumor_slice_index}"
                
            assert pre_slice_pad.shape == post_slice_pad.shape  == (256, 256), f"Shapes do not match: {pre_slice_pad.shape}, {post_slice_pad.shape}"
            label = 0 if has_tumor_cells(mask_slice[0], threshold=self.tumor_sensitivity) else 1
            if label == 0:
                print("Tumor found")
            if slice_has_high_info(pre_slice_pad) and slice_has_high_info(post_slice_pad):
                pre_path = self._save_slice(pre_slice_pad, pat_id, pre_slice_index, 'pre', label)
                post_path = self._save_slice(post_slice_pad, pat_id, post_slice_index, 'post', label)
                tumor_path = self._save_slice(mask_slice[0], pat_id, tumor_slice_index, 'tumor', label)
                self.data.append({"pre_path": pre_path, "post_path": post_path, 
                                  "tumor_path": tumor_path, "label": label, "pat_id": pat_id,
                                  "index_pre": pre_slice_index, "index_post": post_slice_index})

    def _save_slice(self, slice_array: ndarray, pat_id: str, index: Tuple, slice_type: str, label: int):
        """Save the 2D slice as a numpy file and return the file path."""
        brain_axis = self.convert_tuple_to_string(index)
        ##TODO: remove label from this
        filename = f"{pat_id}_slice_{brain_axis}_{slice_type}_label_{label}.npy"
        save_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(save_path):
            np.save(save_path, slice_array)
        return save_path
    
    def convert_tuple_to_string(self, index):
        
        if index[0] != -1:
            return "axial_" + str(index[0])
        elif index[1] != -1:
            return "coronal_" + str(index[1])
        elif index[2] != -1:
            return "sagittal" + str(index[2])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        triplet = self.data[idx]
        pre_slice = np.load(triplet["pre_path"])
        post_slice = np.load(triplet["post_path"])
        baseline = get_baseline_np(pre_slice, post_slice)
        assert pre_slice.shape == post_slice.shape == (256, 256), f"Shapes do not match: {pre_slice.shape}, {post_slice.shape}"
        # tumor_slice = np.load(triplet["tumor_path"]) if "tumor_path" in triplet else None

        # Apply any transformations if necessary
        if self.transform:
            pre_slice = self.transform(pre_slice)
            post_slice = self.transform(post_slice)
            # if tumor_slice is not None:
            #     tumor_slice = self.transform(tumor_slice)

        return {"pre": pre_slice, "post": post_slice, "label": triplet["label"], 
                "pat_id": triplet["pat_id"], "index_pre": triplet["index_pre"], "index_post": triplet["index_post"],
                "baseline": baseline}
        
class aertsDataset(Dataset):
    def __init__(self, proc_preop: str, raw_tumor_dir: str, image_ids: list, save_dir: str, skip:int=1, tumor_sensitivity = 0.10, transform=None):
        self.root = proc_preop
        self.raw_tumor_dir = raw_tumor_dir
        self.transform = transform
        self.image_ids = image_ids
        self.save_dir = save_dir  # Directory to save 2D slices
        self.skip = skip
        self.tumor_sensitivity = tumor_sensitivity
        self.data = []
        # https://www.nature.com/articles/s41597-022-01806-4

        os.makedirs(self.save_dir, exist_ok=True)  # Ensure the save directory exists
        
        for root, dirs, files in os.walk(self.root):
            for filename in files:
                for image_id in self.image_ids:
                    if filename.endswith(image_id):
                        try:
                            pat_id = root.split("/")[-1]
                            print(f"Processing {pat_id}")
                            preop_nifti = nib.load(os.path.join(root, filename))
                            postop_nifti = nib.load(os.path.join(root.replace("preop", "postop"), 
                                                                filename.replace("preop", "postop")))

                            if "PAT" in pat_id:
                                try:
                                    tumor = nib.load(os.path.join(f"{raw_tumor_dir}/{pat_id}/anat/{pat_id}_space_T1_label-tumor.nii"))
                                    tumor_resampled = resample_to_img(tumor, preop_nifti, interpolation='nearest')
                                    tumor_norm = normalize_nifti(tumor_resampled)
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
                                # return
                        except FileNotFoundError as e:
                            print(f"File not found: {e}")
                        except Exception as e:
                            print(f"Uncaught error: {e}")

    def _process_con_slices(self, pat_id, images_pre, images_post):
        """Process control patient (CON) slices and save them."""
        for i, (pre_slice, post_slice) in enumerate(zip(images_pre, images_post)):
            pre_slice_pad = pad_slice(pre_slice[0])
            post_slice_pad = pad_slice(post_slice[0])
            ## post_slice[0] is an image, post_slice[1] is the index
            pre_slice_index: Tuple[int, int, int] = pre_slice[1]
            post_slice_index: Tuple[int, int, int] = post_slice[1]
            
            assert pre_slice_index == post_slice_index, f"Indices tuples do not match: {pre_slice_index}, {post_slice_index}"
            assert pre_slice_pad.shape == post_slice_pad.shape == (256, 256), f"Shapes do not match: {pre_slice_pad.shape}, {post_slice_pad.shape}"
            if slice_has_high_info(pre_slice_pad) and slice_has_high_info(post_slice_pad):
                pre_path = self._save_slice(pre_slice_pad, pat_id, pre_slice_index, 'pre', 1)
                post_path = self._save_slice(post_slice_pad, pat_id, post_slice_index, 'post', 1)
                self.data.append({"pre_path": pre_path, "post_path": post_path, "label": 1, "pat_id": pat_id,
                                  "index_pre": pre_slice_index, "index_post": post_slice_index})

    def _process_pat_slices(self, pat_id, images_pre, images_post, mask_slices):
        """Process patient (PAT) slices and save them."""
        for i, (pre_slice, post_slice, mask_slice) in enumerate(zip(images_pre, images_post, mask_slices)):
            pre_slice_pad = pad_slice(pre_slice[0])
            post_slice_pad = pad_slice(post_slice[0])
            
            pre_slice_index: Tuple[int, int, int] = pre_slice[1]
            post_slice_index: Tuple[int, int, int] = post_slice[1]
            tumor_slice_index: Tuple[int, int, int] = mask_slice[1]
            assert pre_slice_index == post_slice_index == tumor_slice_index, f"Indices\
                tuples do not match: {pre_slice_index}, {post_slice_index}, {tumor_slice_index}"
                
            assert pre_slice_pad.shape == post_slice_pad.shape  == (256, 256), f"Shapes do not match: {pre_slice_pad.shape}, {post_slice_pad.shape}"
            label = 0 if has_tumor_cells(mask_slice[0], threshold=self.tumor_sensitivity) else 1
            
            if slice_has_high_info(pre_slice_pad) and slice_has_high_info(post_slice_pad):
                pre_path = self._save_slice(pre_slice_pad, pat_id, pre_slice_index, 'pre', label)
                post_path = self._save_slice(post_slice_pad, pat_id, post_slice_index, 'post', label)
                tumor_path = self._save_slice(mask_slice[0], pat_id, tumor_slice_index, 'tumor', label)
                self.data.append({"pre_path": pre_path, "post_path": post_path, 
                                  "tumor_path": tumor_path, "label": label, "pat_id": pat_id,
                                  "index_pre": pre_slice_index, "index_post": post_slice_index})

    def _save_slice(self, slice_array: ndarray, pat_id: str, index: Tuple, slice_type: str, label: int):
        """Save the 2D slice as a numpy file and return the file path."""
        brain_axis = self.convert_tuple_to_string(index)
        ##TODO: remove label from this
        filename = f"{pat_id}_slice_{brain_axis}_{slice_type}_label_{label}.npy"
        save_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(save_path):
            np.save(save_path, slice_array)
        return save_path
    
    def convert_tuple_to_string(self, index):
        
        if index[0] != -1:
            return "axial_" + str(index[0])
        elif index[1] != -1:
            return "coronal_" + str(index[1])
        elif index[2] != -1:
            return "sagittal" + str(index[2])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        triplet = self.data[idx]
        pre_slice = np.load(triplet["pre_path"])
        post_slice = np.load(triplet["post_path"])
        baseline = get_baseline_np(pre_slice, post_slice)
        assert pre_slice.shape == post_slice.shape == (256, 256), f"Shapes do not match: {pre_slice.shape}, {post_slice.shape}"
        # tumor_slice = np.load(triplet["tumor_path"]) if "tumor_path" in triplet else None

        # Apply any transformations if necessary
        if self.transform:
            pre_slice = self.transform(pre_slice)
            post_slice = self.transform(post_slice)
            # if tumor_slice is not None:
            #     tumor_slice = self.transform(tumor_slice)

        return {"pre": pre_slice, "post": post_slice, "label": triplet["label"], 
                "pat_id": triplet["pat_id"], "index_pre": triplet["index_pre"], "index_post": triplet["index_post"],
                "baseline": baseline}


                

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

def create_loaders_with_index(dataset, train_index, test_index, batch_size=1):
    train_dataset = Subset(dataset, train_index)
    test_dataset = Subset(dataset, test_index)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def create_loaders_with_split(dataset: Dataset, split=(0.8, 0.2), generator=None):
    train_t1, test_t1 = random_split(dataset=dataset, lengths=split, generator=generator)
    BATCH_SIZE=1
    train_loader_t1 = DataLoader(train_t1, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_t1 = DataLoader(test_t1, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader_t1, test_loader_t1

# class ShiftImage:
#     def __init__(self, max_shift_x=10, max_shift_y=10):
#         self.max_shift_x = max_shift_x
#         self.max_shift_y = max_shift_y
#     def __call__(self, image):
#         # Randomly shift the image
#         shift_x = random.randint(-self.max_shift_x, self.max_shift_x)
#         shift_y = random.randint(-self.max_shift_y, self.max_shift_y)
        
#         # Shift the image using affine transformation

#         # return F.affine(image, angle=0, translate=(shift_x, shift_y), scale=1, shear=0, 
#         #                 interpolation=F.InterpolationMode.NEAREST)

#         return torch.roll(image, shifts=(shift_x, shift_y), dims=(1, 2))
# class ShiftImage:
#     def __init__(self, max_shift_x=10, max_shift_y=10, mode='bilinear', padding_mode='border', align_corners=True):
#         self.max_shift_x = max_shift_x
#         self.max_shift_y = max_shift_y
#         self.mode = mode
#         self.padding_mode = padding_mode
#         self.align_corners = align_corners

#     def __call__(self, image):
#         # Randomly generate shift values for x and y axes
#         shift_x = torch.randint(-self.max_shift_x, self.max_shift_x + 1, (1,)).item()
#         shift_y = torch.randint(-self.max_shift_y, self.max_shift_y + 1, (1,)).item()

#         theta = torch.tensor([[1, 0, shift_x], [0, 1, shift_y]], dtype=torch.float).unsqueeze(0)

#         grid = torch.nn.functional.affine_grid(theta, image.unsqueeze(0).size(), align_corners=self.align_corners)
#         shifted_image = torch.nn.functional.grid_sample(image.unsqueeze(0), grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)

#         return shifted_image.squeeze(0)