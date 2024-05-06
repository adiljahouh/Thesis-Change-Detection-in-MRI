import torchio as tio
import os
from torch.utils.data import DataLoader, Dataset, random_split, Subset

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