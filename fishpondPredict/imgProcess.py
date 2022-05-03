from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.segmentation import find_boundaries
import glob
import tqdm
import os
import gdalTools
from skimage.morphology import square, dilation


def get_labels():
    """Load the mapping that associates classes with label colors

    Returns:
        np.ndarray with dimensions (13, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],
            [0, 200, 0],
            [150, 250, 0],
            [150, 200, 150],
            [200, 0, 200],
            [150, 0, 250],
            [150, 150, 250],
            [250, 200, 0],
            [200, 200, 0],
            [100, 56, 250],
            [250, 150, 56],
            [200, 0, 0],
            [250, 0, 150],
            [200, 150, 150],
            [250, 150, 125],
            [0, 0, 200],
            [0, 100, 50],
            [0, 150, 200],
            [0, 200, 250],
            [255, 255, 255],
            # [255, 255, 0],
            [250, 128, 114],
            [255, 20, 147],
            [105, 139, 34],
            [139, 58, 58],
            [255,69,0],
            [255,165,0],
            [128,128,0],
            [64,224,208]
        ]
    )


def decode_segmap(label_mask, n_classes):
    """Decode segmentation class labels into a color image

    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.

    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_labels()
    color_length = len(label_colours)
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        if ll < color_length:
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        else:
            lll = ll%(color_length-1)
            if lll == 0:
                lll = 13
            r[label_mask == ll] = label_colours[lll, 0]
            g[label_mask == ll] = label_colours[lll, 1]
            b[label_mask == ll] = label_colours[lll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb.astype(np.uint8)


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    imgPath = r'D:\MyWorkSpace\paper\fishpond\fishpondCode\fishpondPredict\predict_ours_model+pointrend\fusion.tif'
    outPath = r'D:\MyWorkSpace\paper\fishpond\fishpondCode\fishpondPredict\predict_ours_model+pointrend\fusion_rgb.tif'
    im_proj, im_geotrans, im_width, im_height, im_data = gdalTools.read_img(imgPath)
    labeled_array, num_features = ndimage.label(im_data)
    decode = decode_segmap(labeled_array, n_classes=num_features)
    gdalTools.write_img(outPath, im_proj, im_geotrans, decode.transpose(2, 0, 1))

