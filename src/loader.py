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
from scipy.ndimage import shift

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

class subject_patient_pairs(Dataset):
    """
    Image dataset for each subject in the dataset
    creating only 'correct' and 'incorrect' pairs for now

    Works by passing preop or postop directory to the class
    and finds the corresponding image in the other dir and labels
    """
    def __init__(self, proc_preop: str, raw_tumor_dir: str, image_ids: list, transform=None, skip:int=1,
                 tumor_sensitivity = 0.10):
        self.root = proc_preop
        self.transform = transform
        self.data = []
        self.image_ids = image_ids
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
                            # print(preop_nifti.shape)
                            # print(postop_nifti.shape)
                            # load the tumor from the tumor directory matching the patient id
                            if "PAT" in pat_id:
                                try:
                                    tumor = nib.load(os.path.join(f"{raw_tumor_dir}/{pat_id}/anat/{pat_id}_space_T1_label-tumor.nii"))
                                    tumor_resampled = resample_to_img(tumor, preop_nifti, interpolation='nearest')
                                    tumor_norm = normalize_nifti(tumor_resampled)
                                    assert tumor_norm.max() <= 1.0, f"max: {tumor_norm.max()}"
                                    assert tumor_norm.min() >= 0.0, f"min: {tumor_norm.min()}"
                                except FileNotFoundError as e:
                                    print(f"Tumor not found for {pat_id}, {e}")
                                except Exception as e:
                                    print(f"Uncaught error, {e}")
                            
                            # resample the postop nifti to the preop nifti
                            preop_nifti_norm = normalize_nifti(preop_nifti)
                            postop_nifti_norm = normalize_nifti(postop_nifti)
                            assert preop_nifti_norm.max() <= 1.0, f"max: {preop_nifti_norm.max()}"
                            assert postop_nifti_norm.min() >= 0.0, f"min: {postop_nifti_norm.min()}"

                            if "-CON" in pat_id:
                                assert preop_nifti_norm.shape == postop_nifti_norm.shape
                                
                                images_pre = convert_3d_into_2d(preop_nifti_norm, skip=skip)
                                images_post = convert_3d_into_2d(postop_nifti_norm, skip = skip)

                                # Create triplets with label 1 (similar slices)
                                images_pre_pad = [(pad_slice(image[0]), image[1], 1) for image in images_pre]
                                images_post_pad = [(pad_slice(image[0]), image[1], 1) for image in images_post]
                                
                                assert len(images_pre_pad) == len(images_post_pad)
                                
                                # Create triplets (pre_slice, post_slice, label, tumor=None)
                                triplets_con = [{"pre": pre, "post": post, "label": label, "tumor": np.zeros_like(pre), 
                                                 "pat_id": pat_id, "index_pre": index_pre, "index_post": index_post} 
                                                for (pre, index_pre, label), 
                                                (post, index_post, _) in 
                                                zip(images_pre_pad, images_post_pad) if 
                                                slice_has_high_info(pre) and slice_has_high_info(post)]
                                self.data.extend(triplets_con)
                            
                            elif "-PAT" in pat_id:
                                assert preop_nifti_norm.shape == postop_nifti_norm.shape == tumor_norm.shape

                                images_pre = convert_3d_into_2d(preop_nifti_norm, skip=skip)
                                images_post = convert_3d_into_2d(postop_nifti_norm, skip=skip)
                                mask_slices = convert_3d_into_2d(tumor_norm, skip=skip)
                                # Create triplets with label 0 if the slice contains a tumor
                                images_pre_pad = [(pad_slice(image[0]), image[1], 0 if has_tumor_cells(mask_slice[0], threshold=tumor_sensitivity) else 1) for image, mask_slice in zip(images_pre, mask_slices)]
                                images_post_pad = [(pad_slice(image[0]), image[1], 0 if has_tumor_cells(mask_slice[0], threshold=tumor_sensitivity) else 1) for image, mask_slice in zip(images_post, mask_slices)]
                                # pad the tumor mask as well
                                mask_slices_pad = [(pad_slice(mask_slice[0]), mask_slice[1]) for mask_slice in mask_slices]
                                assert len(images_pre_pad) == len(images_post_pad) == len(mask_slices_pad)
                                
                                # Create triplets (pre_slice, post_slice, label, tumor)
                                triplets_pat = [{"pre": pre, "post": post, "label": label, "tumor": mask_slice, 
                                                 "pat_id": pat_id, "index_pre": index_pre, "index_post": index_post} 
                                                for (pre, index_pre, label), (post, index_post, _), (mask_slice, _) in 
                                                zip(images_pre_pad, images_post_pad, mask_slices_pad) if 
                                                slice_has_high_info(pre) and slice_has_high_info(post)]
                                
                                self.data.extend(triplets_pat)
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
            item = self.data[idx]
            try:
                item['pre'] = self.transform(item["pre"]).cpu().numpy().squeeze(0)
                item['post'] = self.transform(item["post"]).cpu().numpy().squeeze(0)
            except Exception:
                print("Couldnt transform and return an array, be sure you also pass ToTensor to the transform")
        return self.data[idx]
                
class shifted_subject_patient_pairs(Dataset):
    """
    Image dataset for each subject in the dataset transformed by a random shift
    creating only 'correct' and 'incorrect' pairs for now

    Works by passing preop or postop directory to the class
    and finds the corresponding image in the other dir and labels
    """
    def __init__(self, proc_preop: str, raw_tumor_dir: str, image_ids: list, transform=None, skip:int=1,
                 tumor_sensitivity = 0.10):
        self.root = proc_preop
        self.transform = transform
        self.data = []
        self.image_ids = image_ids
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
                                    assert tumor_norm.max() <= 1.0, f"max: {tumor_norm.max()}"
                                    assert tumor_norm.min() >= 0.0, f"min: {tumor_norm.min()}"
                                except FileNotFoundError as e:
                                    print(f"Tumor not found for {pat_id}, {e}")
                                except Exception as e:
                                    print(f"Uncaught error, {e}")
                            
                            # resample the postop nifti to the preop nifti
                            preop_nifti_norm = normalize_nifti(preop_nifti)
                            postop_nifti_norm = normalize_nifti(postop_nifti)
                            assert preop_nifti_norm.max() <= 1.0, f"max: {preop_nifti_norm.max()}"
                            assert postop_nifti_norm.min() >= 0.0, f"min: {postop_nifti_norm.min()}"
                            shift_values = (random.randint(0, 50), random.randint(0, 50))
                            if "-CON" in pat_id:
                                assert preop_nifti_norm.shape == postop_nifti_norm.shape
                                
                                images_pre = convert_3d_into_2d(preop_nifti_norm, skip=skip)
                                images_post = convert_3d_into_2d(postop_nifti_norm, skip = skip)

                                # Create triplets with label 1 (similar slices)
                                images_pre_pad = [(pad_slice(image[0]), image[1], 1) for image in images_pre]
                                images_post_pad = [(pad_slice(image[0]), image[1], 1) for image in images_post]
                                
                                assert len(images_pre_pad) == len(images_post_pad)
                                
                                # Create triplets (pre_slice, post_slice, label, tumor=None)
                                
                                triplets_con = [{"pre": shift_image_numpy(pre, shift_amount=shift_values), 
                                                 "post": shift_image_numpy(post, shift_amount=shift_values),
                                                  "label": label, "tumor": np.zeros_like(pre), 
                                                 "pat_id": pat_id, "index_pre": index_pre, "index_post": index_post} 
                                                for (pre, index_pre, label), (post, index_post, _) in 
                                                zip(images_pre_pad, images_post_pad) if 
                                                slice_has_high_info(pre) and slice_has_high_info(post)]
                                self.data.extend(triplets_con)
                            
                            elif "-PAT" in pat_id:
                                assert preop_nifti_norm.shape == postop_nifti_norm.shape == tumor_norm.shape

                                images_pre = convert_3d_into_2d(preop_nifti_norm, skip=skip)
                                images_post = convert_3d_into_2d(postop_nifti_norm, skip=skip)
                                mask_slices = convert_3d_into_2d(tumor_norm, skip=skip)
                                # Create triplets with label 0 if the slice contains a tumor
                                images_pre_pad = [(pad_slice(image[0]), image[1], 0 if has_tumor_cells(mask_slice[0], threshold=tumor_sensitivity) else 1) for image, mask_slice in zip(images_pre, mask_slices)]
                                images_post_pad = [(pad_slice(image[0]), image[1], 0 if has_tumor_cells(mask_slice[0], threshold=tumor_sensitivity) else 1) for image, mask_slice in zip(images_post, mask_slices)]
                                # pad the tumor mask as well
                                mask_slices_pad = [(pad_slice(mask_slice[0]), mask_slice[1]) for mask_slice in mask_slices]
                                assert len(images_pre_pad) == len(images_post_pad) == len(mask_slices_pad)
                            
                                # Create triplets (pre_slice, post_slice, label, tumor)
                                triplets_pat = [{"pre": shift_image_numpy(pre, shift_amount=shift_values), 
                                                 "post": shift_image_numpy(post, shift_amount=shift_values),
                                                 "label": label, "tumor": mask_slice, 
                                                 "pat_id": pat_id, "index_pre": index_pre, 
                                                 "index_post": index_post} 
                                                for (pre, index_pre, label), (post, index_post, _), (mask_slice, _) in 
                                                zip(images_pre_pad, images_post_pad, mask_slices_pad) if 
                                                slice_has_high_info(pre) and slice_has_high_info(post)]
                                 
                                self.data.extend(triplets_pat)
                        except FileNotFoundError as e:
                            print(f"{e}, this is normal to happen for 3 subjects which have no postoperative data")
                        except Exception as e:
                            print(f"Uncaught error, {e}")
                    else:
                        pass
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]               
    

                

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