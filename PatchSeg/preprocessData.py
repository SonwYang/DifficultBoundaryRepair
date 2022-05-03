import gdalTools
import os
import glob
import numpy as np
from skimage.morphology import dilation, square
from scipy.ndimage.morphology import *
import cv2
import shutil
from osgeo import gdal
import warnings
warnings.filterwarnings("ignore")

def normal_img(img):
    # low_thresold = np.percentile(img, 5)
    # high_thresold = np.percentile(img, 95)
    # img = np.clip(img, low_thresold, high_thresold)
    min = np.min(img)
    max = np.max(img)
    img = (img - min) / (max - min) * 255
    return img.astype(np.uint8)


def read_geoimg(path):
    data = gdal.Open(path)
    lastChannel = data.RasterCount + 1
    arr = [normal_img(data.GetRasterBand(idx).ReadAsArray()) for idx in range(1, lastChannel)]
    arr = np.dstack(arr)
    return arr.transpose(2, 0, 1).squeeze()


def prepareData(dataRoot):
    ### Config
    # dataRoot = r'D:\MyWorkSpace\paper\fishpond\data\data'
    outRoot = 'data'
    gdalTools.mkdir(outRoot)
    imageRoot = os.path.join(outRoot, 'train_images')
    gdalTools.mkdir(imageRoot)
    labelRoot = os.path.join(outRoot, 'train_labels')
    gdalTools.mkdir(labelRoot)
    gaussianRoot = os.path.join(outRoot, 'train_gaussian')
    gdalTools.mkdir(gaussianRoot)
    polyRoot = os.path.join(outRoot, 'train_polys')
    gdalTools.mkdir(polyRoot)
    lineRoot = os.path.join(outRoot, 'train_edges')
    gdalTools.mkdir(lineRoot)

    polyList = glob.glob(f'{dataRoot}/*/*POLY.shp')
    for PolyPath in polyList:
        LinePath = PolyPath.replace('POLY', 'LINE')
        if os.path.exists(LinePath):
            continue
        gdalTools.pol2line(PolyPath, LinePath)

    for PolyPath in polyList:
        LinePath = PolyPath.replace('POLY', 'LINE')
        baseName = os.path.basename(PolyPath).split('_')[0]
        imgPath = os.path.split(PolyPath)[0] + f'/{baseName}_Feature.tif'
        assert os.path.exists(imgPath), f"please check your data path:{imgPath}"

        im_proj, im_geotrans, im_width, im_height, im_data = gdalTools.read_img(imgPath)
        img = read_geoimg(imgPath)

        rasterLinePath = os.path.join(lineRoot, baseName + '.tif')
        rasterPolyPath = os.path.join(polyRoot, baseName + '.tif')
        rasterImgPath = os.path.join(imageRoot, baseName + '.tif')

        gdalTools.shp2Raster(LinePath, imgPath, rasterLinePath, nodata=0)
        gdalTools.shp2Raster(PolyPath, imgPath, rasterPolyPath, nodata=0)
        # shutil.copyfile(imgPath, rasterImgPath)
        gdalTools.write_img(rasterImgPath, im_proj, im_geotrans, img)

    LineList = glob.glob(f'{lineRoot}/*.tif')
    classList = [0, 1, 2, 4, 20, 34, 93, 154, 255]

    for linePath in LineList:
        polyPath = linePath.replace('edges', 'polys')
        im_proj, im_geotrans, im_width, im_height, line = gdalTools.read_img(linePath)
        _, _, _, _, poly = gdalTools.read_img(polyPath)
        line = np.array(line)
        line = np.where(line > 0, 1, 0)
        poly = np.array(poly)
        poly = np.where(poly > 0, 1, 0)

        line2 = dilation(line.copy(), square(2))
        label = line2.copy()
        label = np.where(line2 > 0, 3, poly)

        labelPath = linePath.replace('edges', 'labels')
        cv2.imwrite(labelPath, label)

        # creation of gaussian masks
        distance_array = distance_transform_edt(1 - line)
        std = 1
        distance_array = np.exp(-0.5 * (distance_array * distance_array) / (std * std))
        distance_array *= 255
        distance_array = distance_array.astype(np.uint8)
        gaussian_mask = np.zeros_like(distance_array)
        for i in range(len(classList)):
            # for j in classList[i]:
            gaussian_mask = np.where(distance_array == classList[i], i, gaussian_mask)

        gaussianPath = linePath.replace('edges', 'gaussian')
        cv2.imwrite(gaussianPath, gaussian_mask)

if __name__ == '__main__':
    prepareData(r'D:\BaiduNetdiskDownload\GF3_Yangzhitang_Samples_Feature')
    # imgPath = r'D:\MyWorkSpace\paper\fishpond\data\tempData\train_images\00000001.tif'
    # img = read_geoimg(imgPath).transpose(1, 2, 0)
    # data = cv2.copyMakeBorder(img, 0, 100, 0, 100, cv2.BORDER_REFLECT)
    # print(data.shape)

