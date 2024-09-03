import torch.nn as nn
import torch
import numpy as np
import cv2
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from distance_measures import various_distance
def merge_images(image1, image2, image3, output_path):
    # Create a new figure
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Display each image on a separate subplot
    axs[0].imshow(image1, cmap="gray")
    axs[0].axis('off')
    axs[1].imshow(image2, cmap="gray")
    axs[1].axis('off')
    axs[2].imshow(image3)
    axs[2].axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the merged image
    plt.savefig(output_path)
    plt.close(fig)
    
def generate_roc_curve(distances, labels, save_dir):
    # # Invert distances because lower distance indicates more similarity
    distances = [d.cpu().item() for d in distances]
    labels = [l.cpu().item() for l in labels]

    scores = [-d for d in distances]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the plot to the specified directory
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()  # Close the plot to free up memory
    return thresholds
def single_layer_similar_heatmap_visual(output_t0,output_t1,dist_flag):

    interp = nn.Upsample(size=[512,512], mode='bilinear')
    c, h, w = output_t0.data.shape
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    # print(distance)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()
    similar_distance_map_rz = interp(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :]))
    # print(similar_distance_map_rz)
    similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    return similar_dis_map_colorize, 255*similar_distance_map_rz.data.cpu().numpy()[0][0]

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
    
## TODO: compute request kirsten
## TODO: nifti output? pass header and affine
## TODO:  https://medium.com/@rehman.aimal/implement-3d-unet-for-cardiac-volumetric-mri-scans-in-pytorch-79f8cca7dc68

## DONE: Try slicer, models seem to all not work (glioma and meninglomas) on pat 11 and 23 which are obvious
## switched to padding but need compute cuz the images are just too much data if pad them all (reserves affine) 
## requested compute
## imported vgg16 and trying it as its a better architecture for similarity learning - need compute
## Using roc curves now but running into issue of decision line with low test numbers
## read (https://pure.tue.nl/ws/portalfiles/portal/292941365/Trinh_P.pdf)
## thinking about adding pre and pre and post post but there would be a class mismatch
## Tried Monai, segmentation

## Ask maxime if the data went through the ethical review board or data steward was informed (floor lup)
## problem: low data,    need to find a way to increase data
## best way: voxel level with labels (Slicer, ITK-snap, MITK, MRIcon)
## maybe already train with dissimilarities from other diseases if we have more data there?

    
## unet and other link
#cross validation
def multiple_layer_similar_heatmap_visiual(output_t0, output_t1, dist_flag):
    # Assuming output_t0 and output_t1 are torch tensors of shape (n, c, d, h, w)
    interp = nn.Upsample(size=[164,164, 164], mode='trilinear')   
    n, c, d, h, w = output_t0.shape
    
    # Reshape to (n, c, d*h*w)
    out_t0_flat = output_t0.view(c, d*h*w).transpose(1, 0)
    out_t1_flat = output_t1.view(c, d*h*w).transpose(1, 0)
    # Compute distance using a predefined function
    distance = various_distance(out_t0_flat, out_t1_flat, dist_flag=dist_flag)
    similar_distance_map = distance.view(d,h,w).data.cpu().numpy()
    # print(similar_distance_map.shape)
    similar_distance_map_rz = interp(torch.from_numpy(similar_distance_map).unsqueeze(0).unsqueeze(0))
    scaled_data =  similar_distance_map_rz.data.cpu().numpy()[0][0] *255
    # grid = pyvista.ImageData(dimensions=np.array(scaled_data.shape)+1)  # Create an empty grid()
    # grid.cell_data["Similarity"]= scaled_data.flatten(order="F")
    # # Create a plotter object and set the grid as the active scalar field
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(grid, scalars="Similarity")
    # plotter.show_grid()
    # plotter.show()
    # similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    sliced_images = [scaled_data[i] for i in range(scaled_data.shape[0])]
    
    return sliced_images

