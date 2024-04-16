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


def multiple_layer_similar_heatmap_visiual(output_t0, output_t1, dist_flag):
    # Assuming output_t0 and output_t1 are torch tensors of shape (n, c, d, h, w)
    interp = nn.Upsample(size=[512,512, 512], mode='trilinear')
    n, c, d, h, w = output_t0.shape
    print(n, c, d, h, w)
    
    # Reshape to (n, c, d*h*w)
    out_t0_flat = output_t0.view(c, d*h*w).transpose(1, 0)
    out_t1_flat = output_t1.view(c, d*h*w).transpose(1, 0)
    print(out_t0_flat.shape, out_t1_flat.shape)
    # Compute distance using a predefined function
    distance = various_distance(out_t0_flat, out_t1_flat, dist_flag=dist_flag)
    print(distance.shape)
    similar_distance_map = distance.view(d,h,w).data.cpu().numpy()
    # print(similar_distance_map.shape)
    similar_distance_map_rz = interp(torch.from_numpy(similar_distance_map).unsqueeze(0).unsqueeze(0))
    print(similar_distance_map_rz.shape)
    # print(similar_distance_map_rz)
    grid = pyvista.ImageData(dimensions=np.array(similar_distance_map_rz.shape)+1)  # Create an empty grid()
    grid.cell_data["Similarity"] = similar_distance_map_rz.data.cpu().numpy()[0][0]

    # Create a plotter object and set the grid as the active scalar field
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, scalars="Similarity", cmap="bone")
    plotter.show_grid()
    plotter.show()
    sleep(10)
    # similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    return "test"
