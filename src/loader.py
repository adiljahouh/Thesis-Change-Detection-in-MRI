import torchio as tio
import os
import nibabel as nib
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from nilearn.image import resample_to_img
from numpy import ndarray
from typing import Tuple
import numpy as np
def normalize_nifti(nifti_image: nib.Nifti1Image) -> ndarray:
    return (nifti_image.get_fdata() - np.min(nifti_image.get_fdata())) / (np.max(
        nifti_image.get_fdata()) - np.min(nifti_image.get_fdata()))

def pad_slice(slice_2d, output_size=(256, 256)):
    """
    Pad a 2D slice to the desired output size with zeros.
    """
    pad_height = (output_size[0] - slice_2d.shape[0])
    pad_width = (output_size[1] - slice_2d.shape[1])
    
    padded_slice = np.pad(slice_2d, 
                          ((0, pad_height), (0, pad_width)), 
                          mode='constant', 
                          constant_values=0)
    return padded_slice

def convert_3d_into_2d(nifti_image: ndarray) -> list[ndarray]:
    slices = []
   
    # (axial)
    ## TODO: Use all slices for now only using every 4th slice
    for i in range(nifti_image.shape[0]):
        if i % 4 == 0:
            slices.append(nifti_image[i, :, :]) 
    #  (coronal)
    for i in range(nifti_image.shape[1]):
        if i % 4 == 0:
            slices.append(nifti_image[:, i, :])  
    # (sagittal)
    for i in range(nifti_image.shape[2]):
        if i % 4 == 0:
            slices.append(nifti_image[:, :, i])
    
    return slices

def check_tumor_presence(slice_2d: ndarray, threshold=1):
    return np.any(slice_2d > threshold)

class imagePairs(Dataset):
    """
    Image dataset for each subject in the dataset
    creating only 'correct' and 'incorrect' pairs for now

    Works by passing preop or postop directory to the class
    and finds the corresponding image in the other dir and labels
    """
    def __init__(self, proc_preop: str, raw_tumor_dir: str, image_ids: list, transform=None):
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
                            # load the tumor from the tumor directory matching the patient id
                            if "PAT" in pat_id:
                                try:
                                    tumor = nib.load(os.path.join(f"{raw_tumor_dir}/{pat_id}/anat/{pat_id}_space_T1_label-tumor.nii"))
                                    tumor_resampled = resample_to_img(tumor, preop_nifti, interpolation='nearest')
                                    tumor_norm = normalize_nifti(tumor_resampled)
                                except FileNotFoundError as e:
                                    print(f"Tumor not found for {pat_id}, {e}")
                                except Exception as e:
                                    print(f"Uncaught error, {e}")
                            
                            # resample the postop nifti to the preop nifti
                            preop_nifti_norm = normalize_nifti(preop_nifti)
                            postop_nifti_norm = normalize_nifti(postop_nifti)

                            if "-CON" in pat_id:
                                images_pre: list = convert_3d_into_2d(preop_nifti_norm)
                                images_post: list = convert_3d_into_2d(postop_nifti_norm)

                                images_pre_pad = [(pad_slice(image), 1) for image in images_pre]
                                images_post_pad = [(pad_slice(image), 1) for image in images_post]
                                triplets_con = [(pre, post, label) for (pre, _), (post, label) in zip(images_pre_pad, images_post_pad)]
                                self.data.extend(triplets_con)
                            elif "-PAT" in pat_id:
                                images_pre: list = convert_3d_into_2d(preop_nifti_norm)
                                images_post: list = convert_3d_into_2d(postop_nifti_norm)
                                mask_slices = convert_3d_into_2d(tumor_norm)

                                # Create triplets with label 0 if the slice contains a tumor
                                images_pre_pad = [(pad_slice(image), check_tumor_presence(mask_slice)) for image, mask_slice in zip(images_pre, mask_slices)]
                                images_post_pad = [(pad_slice(image), check_tumor_presence(mask_slice)) for image, mask_slice in zip(images_post, mask_slices)]

                                # Create triplets (pre_slice, post_slice, label)
                                triplets_pat = [(pre, post, label) for (pre, label), (post, _) in zip(images_pre_pad, images_post_pad)]  
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
            pass
            # img1_file = self.transform(self.data[idx][0])
            # img2_file = self.transform(self.data[idx][1])
        return self.data[idx]

test = imagePairs(proc_preop='./data/processed/preop/BTC-preop', 
                  raw_tumor_dir='./data/raw/preop/BTC-preop/derivatives/tumor_masks',
                  image_ids=['t1_ants_aligned.nii.gz'])
print(test.__len__())
def create_voxel_pairs(proc_preop: str, raw_tumor_dir: str, image_ids: list) -> list[tuple]:
    """
        Proc_preop:  should be the preoperative directory with all patients dirs
        e.g ./data/processed/preop/BTC-preop
        
        raw_tumor_dir: should be the raw tumor directory with all patients dirs
        e.g ./data/raw/preop/BTC-preop/derivatives/tumor_masks

        image_id: list of the image id to match the preop and postop images e.g. t1_ants_aligned.nii.gz

    """

    voxel_data: list[tuple] = []
    for root, dirs, files in os.walk(proc_preop):
        for filename in files:
            for image_id in image_ids:
                if filename.endswith(image_id):
                    try:
                        pat_id = root.split("/")[-1]
                        print(f"Processing {pat_id}")
                        preop_nifti = nib.load(os.path.join(root, filename))
                        postop_nifti = nib.load(os.path.join(root.replace("preop", "postop"), 
                                                         filename.replace("preop", "postop")))
                        # load the tumor from the tumor directory matching the patient id
                        if "PAT" in pat_id:
                            try:
                                tumor = nib.load(os.path.join(f"{raw_tumor_dir}/{pat_id}/anat/{pat_id}_space_T1_label-tumor.nii"))
                                tumor_resampled = resample_to_img(tumor, preop_nifti, interpolation='nearest')
                                tumor_voxels = tumor_resampled.get_fdata()
                                tumor_voxels_norm = tumor_voxels - np.min(tumor_voxels) / (np.max(tumor_voxels) - np.min(tumor_voxels))
                            except FileNotFoundError as e:
                                print(f"Tumor not found for {pat_id}, {e}")
                            except Exception as e:
                                print(f"Uncaught error, {e}")
                        
                        # resample the postop nifti to the preop nifti
                        preop_nifti_norm = preop_nifti.get_fdata() - np.min(preop_nifti.get_fdata()) / (np.max(preop_nifti.get_fdata()) - np.min(preop_nifti.get_fdata()))
                        postop_nifti_norm = postop_nifti.get_fdata() - np.min(postop_nifti.get_fdata()) / (np.max(postop_nifti.get_fdata()) - np.min(postop_nifti.get_fdata()))

                        ## first check if skipping zero and lower voxels is smart by visualizing it

                        if "-CON" in pat_id:
                            for i in range(preop_nifti_norm.shape[0]):
                                for j in range(preop_nifti_norm.shape[1]):
                                    for k in range(preop_nifti_norm.shape[2]):
                                        voxel_data.append(
                                            (preop_nifti_norm[i, j, k], postop_nifti_norm[i, j, k], 1)
                                        )
                        elif "-PAT" in pat_id:
                                for i in range(preop_nifti_norm.shape[0]):
                                    for j in range(preop_nifti_norm.shape[1]):
                                        for k in range(preop_nifti_norm.shape[2]):
                                            if tumor_voxels_norm[i, j, k] > 0.2:
                                                voxel_data.append(
                                                    (preop_nifti_norm[i, j, k], postop_nifti_norm[i, j, k], 0)
                                                )
                                            else:
                                                voxel_data.append(
                                                    (preop_nifti_norm[i, j, k], postop_nifti_norm[i, j, k], 1)
                                                )   
                    except FileNotFoundError as e:
                        print(f"{e}, this is normal to happen for 3 subjects which have no postoperative data")

                    except Exception as e:
                        print(f"Uncaught error, {e}")
                else:
                    pass
    return voxel_data

                                    
def create_subject_pairs(root, id):
    data = []
    for root, dirs, files in os.walk(root):
        for filename in files:
            for image_id in id:
                if filename.endswith(image_id):
                    nifti_1 = tio.ScalarImage(os.path.join(root, filename))
                    try:
                        if "preop" in root:
                            nifti_2 = tio.ScalarImage(os.path.join(root.replace("preop", "postop"), filename.replace("preop", "postop")))
                        else:
                            nifti_2 = tio.ScalarImage(os.path.join(root.replace("postop", "preop"), filename.replace("postop", "preop")))
                        if "-CON" in filename or "-CON" in os.path.join(root, filename):
                            # print("control for ", filename)
                            data.append(
                                tio.Subject(
                                    t1=nifti_1,
                                    t2=nifti_2,
                                    label=1,
                                    name= root.split("/")[-1],
                                    path= os.path.join(root, filename)
                                            )
                                    )
                        elif "-PAT" in filename or "-PAT" in os.path.join(root, filename):
                                data.append(
                                tio.Subject(
                                    t1=nifti_1,
                                    t2=nifti_2,
                                    label=0,
                                    name= root.split("/")[-1],
                                    path= os.path.join(root, filename)
                                            )
                                    )
                        else:
                            print(f"Invalid filename: {os.path.join(root, filename)}")
                    except FileNotFoundError:
                        print(f"Matching subject (pre and post) not found for {os.path.join(root, filename)}")
    return data



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