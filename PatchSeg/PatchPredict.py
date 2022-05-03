
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import gdalTools
import cv2
import numpy as np
from geoClip import *
import tqdm

import matplotlib.pyplot as plt


def point_sparse(pointlist, thresh=12):
    order = pointlist.copy()
    pointKeep = []
    while len(order) > 0:
        point = order[0]
        otherPoints = order[1:]
        keep = compute_euc(otherPoints, point, thresh)
        if keep == 0:
            pointKeep.append(point)
        order.pop(0)
    return pointKeep


def detect_hard_edges(img, step=8):
    r, c = img.shape
    new_image = np.zeros((r, c))
    detection_operator = np.array([[1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1]])
    hard_edges_points = []
    for i in range(32, r-32, step):
        for j in range(32, c-32, step):
            val = np.sum(img[i:i + 7, j:j + 7] * detection_operator)
            pointVal = img[i+4, j+4]
            if val > 10:
                new_image[i+4, j+4] = 1
                hard_edges_points.append([i+4, j+4])
            else:
                new_image[i+1, j+1] = 0
    return np.uint8(new_image), hard_edges_points


def compute_euc(pointlist, point, thresh=12):
    if len(pointlist) > 0:
        distances = np.sqrt(np.sum(np.asarray(np.array(point) - np.array(pointlist)) ** 2, axis=1))
        keep = np.where(distances < thresh, 1, 0).sum()
        return keep
    else:
        return 0


def normal_img(img):
    min = np.min(img)
    max = np.max(img)
    img = (img - min) / (max - min) * 255
    return img.astype(np.uint8)


def inference(point, im_data, model):
    x, y = point
    subImg = im_data[:, x - half_width:x + half_width, y - half_width:y + half_width]
    vis = subImg.transpose((1, 2, 0))[..., :3]
    img = np.array(subImg, dtype=np.float32)
    img /= 255.

    tensor = torch.from_numpy(np.array([img])).cuda()
    outs = model(tensor).cpu().detach()
    prob = outs[0]
    _, maxprob = torch.max(prob, 0)
    maxprob = maxprob.numpy()[12:half_width * 2 - 12, 12:half_width * 2 - 12]
    return maxprob



def get_labels():
    """Load the mapping that associates classes with label colors

    Returns:
        np.ndarray with dimensions (13, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],
            [255, 85, 0],
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

# python PatchPredict.py --imgPath entropy_shannon_subset_Feature.tif --CoarseSegPath coarse.tif --modelPath patch.pth --outPath final.tif
if __name__ == '__main__':
    import argparse
    from preprocessData import prepareData
    parser = argparse.ArgumentParser(
        description='''This is a code for training model.''')
    parser.add_argument('--imgPath', type=str, default=r'D:\BaiduNetdiskDownload\GF3_Yangzhitang_Samples_Feature', help='path of image')
    parser.add_argument('--CoarseSegPath', type=str, default=r'D:\BaiduNetdiskDownload\GF3_Yangzhitang_Samples_Feature', help='path of coarse results')
    parser.add_argument('--modelPath', type=str, default=r'D:\BaiduNetdiskDownload\GF3_Yangzhitang_Samples_Feature', help='path of patch model')
    parser.add_argument('--outPath', type=str, default=r'D:\BaiduNetdiskDownload\GF3_Yangzhitang_Samples_Feature', help='path of out image')
    args = parser.parse_args()

    imgPath = r'D:\MyWorkSpace\paper\fishpond\fishpondCode\fishpondPredict\test\entropy_shannon_subset_Feature.tif'
    CoarseSegPath = r'D:\MyWorkSpace\paper\fishpond\fishpondCode\fishpondPredict\predict_ours_model+pointrend\pointrend.tif'
    modelPath = './ckpts_fishunet/epoch644_model.pth'
    outPath = 'pointrend_v4.tif'

    half_width = 32
    half_width2 = half_width - 24

    model = torch.load(modelPath, map_location="cpu").cuda()
    im_proj, im_geotrans, _, _, im_data = gdalTools.read_img(imgPath)
    arr = [normal_img(im_data[idx]) for idx in range(len(im_data))]
    im_data = np.dstack(arr).transpose(2, 0, 1)
    _, _, _, _, coarseImg = gdalTools.read_img(CoarseSegPath)
    result = coarseImg.copy()
    coarseImg = np.where(coarseImg == 1, 1, 0).astype(np.uint8)
    visImg = np.array(decode_segmap(coarseImg, 2))
    visImg2 = visImg.copy()
    im_width, im_height = coarseImg.shape

    img_vis = np.zeros_like(coarseImg)
    PFMask = np.zeros_like(coarseImg)
    img_vis = [img_vis for _ in range(3)]
    img_vis = np.dstack(img_vis)
    contours, hierarchy = cv2.findContours(coarseImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_vis2 = cv2.drawContours(img_vis.copy(), contours, -1, (255, 255, 255), 1)

    mask = np.where(img_vis2[:, :, 0] > 0, 1, 0)
    _, hard_points = detect_hard_edges(mask, 3)
    hard_points2 = point_sparse(hard_points, 12)
    print(f'nms before:{len(hard_points)}   after:{len(hard_points2)}')

    for i, point in tqdm.tqdm(enumerate(hard_points2)):
        x, y = point
        subImg = im_data[:, x-half_width:x+half_width, y-half_width:y+half_width]
        PFMask[x-half_width2:x+half_width2, y-half_width2:y+half_width2] = 1
        vis = subImg.transpose((1, 2, 0))[..., :3]
        img = np.array(subImg, dtype=np.float32)
        img /= 255.

        tensor = torch.from_numpy(np.array([img])).cuda()
        outs = model(tensor).cpu().detach()
        prob = outs[0]
        _, maxprob = torch.max(prob, 0)
        maxprob = maxprob.numpy()[12:half_width * 2 - 12, 12:half_width * 2 -12]

        result[x-half_width+12:x+half_width-12, y-half_width+12:y+half_width-12] = maxprob

    gdalTools.write_img(outPath, im_proj, im_geotrans, result)


