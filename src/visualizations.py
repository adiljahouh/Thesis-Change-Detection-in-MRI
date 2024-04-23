import torch.nn as nn
import torch
import numpy as np
import cv2
import pyvista
from distance_measures import various_distance
from time import sleep

def single_layer_similar_heatmap_visual(output_t0,output_t1,dist_flag):

    interp = nn.Upsample(size=[512,512], mode='bilinear')
    n, c, h, w = output_t0.data.shape
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    # print(distance)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()
    similar_distance_map_rz = interp(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :]))
    # print(similar_distance_map_rz)
    similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    return similar_dis_map_colorize

## TODO: compute request kirsten
## TODO: pad instead of resize
## TODO: nifti output? pass header and affine
## TODO:  gvgvgg

## DONE: Try slicer, models seem to all not work (glioma and meninglomas)
## switched to padding 
def multiple_layer_similar_heatmap_visiual(output_t0, output_t1, dist_flag):
    # Assuming output_t0 and output_t1 are torch tensors of shape (n, c, d, h, w)
    interp = nn.Upsample(size=[256,256, 256], mode='trilinear')
    
    ## remove this
    # output_t0 = torch.zeros((1, 1, 10, 10, 10))
    # output_t1 = torch.zeros_like(output_t0)
    # output_t0[0, 0, 0, 0, 0] = 1  # Set a single point to a high value

    ##


    n, c, d, h, w = output_t0.shape
    print(n, c, d, h, w)
    
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
    sleep(10)
    # similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    return scaled_data
