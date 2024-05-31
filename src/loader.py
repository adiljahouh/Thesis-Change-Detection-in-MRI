import torchio as tio
import os
import nibabel as nib
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from nilearn.image import resample_to_img
import numpy as np
def make_voxel_pairs(proc_preop: str, raw_tumor_dir: str, image_ids: list):
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
                        
                        if "CON" in pat_id:
                            voxel_data.append(
                                (nifti_1, 1)
                            )
                        elif "-PAT" in filename or "-PAT" in os.path.join(root, filename):
                                voxel_data.append(

                                    )
                    except FileNotFoundError as e:
                        print(f"{e}, this is normal to happen for 3 subjects which have no postoperative data")

                    except Exception as e:
                        print(f"Uncaught error, {e}")
                else:
                    pass
    return data


make_voxel_pairs(proc_preop= './data/processed/preop/BTC-preop', 
                 raw_tumor_dir='./data/raw/preop/BTC-preop/derivatives/tumor_masks', 
                 image_ids=['t1_ants_aligned.nii.gz'])
                                    
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