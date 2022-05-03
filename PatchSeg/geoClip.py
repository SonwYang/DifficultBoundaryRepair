import glob
import os
from tqdm import tqdm
import argparse
import tqdm
from osgeo import gdal
import cv2 as cv
import numpy as np
from PIL import Image
from skimage import io
import gdalTools


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def subImg(img, i, j, targetSize, PaddingSize, height, width):
    if (i + 1) * targetSize < height and (j + 1) * targetSize < width:
        temp_img = img[targetSize * i: targetSize * i + targetSize + PaddingSize,
                   targetSize * j: targetSize * j + targetSize + PaddingSize, :]
        start_x = targetSize * i
        start_y = targetSize * j
    elif (i + 1) * targetSize < height and (j + 1) * targetSize > width:
        temp_img = img[targetSize * i: targetSize * i + targetSize + PaddingSize,
                   width - targetSize - PaddingSize: width, :]
        start_x = targetSize * i
        start_y = width - targetSize - PaddingSize
    elif (i + 1) * targetSize > height and (j + 1) * targetSize < width:
        temp_img = img[height - targetSize - PaddingSize: height,
                   targetSize * j: targetSize * j + targetSize + PaddingSize, :]
        start_x = height - targetSize - PaddingSize
        start_y = targetSize * j
    else:
        temp_img = img[height - targetSize - PaddingSize: height, width - targetSize - PaddingSize: width, :]
        start_x = height - targetSize - PaddingSize
        start_y = width - targetSize - PaddingSize
    return temp_img, (start_x, start_y)

def crop(imgRoot, outRoot, targetSize, PaddingSize, ImgSuffix):
    labels_list = glob.glob(f"./{imgRoot}/*.tif")
    # labels_list = ['./data/train2.tif']

    imgs_num = len(labels_list)
    print("imgs_num:{}".format(labels_list))
    make_dir(outRoot)

    for k in tqdm.tqdm(range(imgs_num)):
        # label = cv.imread(labels_list[k])
        im_proj, im_geotrans, im_width, im_height, label = gdalTools.read_img(labels_list[k])
        label = np.transpose(label, (1, 2, 0))
        label = gdalTools.stretch_n(label, 0, 255, lower_percent=5, higher_percent=95)
        # print(f'max value: {np.max(label)}')
        imgName = os.path.split(labels_list[k])[-1].split('.')[0]

        height, width = label.shape[0], label.shape[1]
        rows, cols = height // targetSize + 1, width // targetSize + 1
        subImg_num = 0
        for i in range(rows):
            for j in range(cols):

                temp_label, start_point = subImg(label, i, j, targetSize, PaddingSize, height, width)
                size = targetSize+PaddingSize
                start_point = (start_point[1] + size // 2, start_point[0] + size // 2)
                tempName = imgName + "_" + str(subImg_num) + ImgSuffix
                tempPath = os.path.join(outRoot, tempName)
                try:
                    gen_geoClips(labels_list[k], tempPath, start_point, size=size)
                    subImg_num += 1
                except:
                    continue


def imagexy2geo(dataset, start_point):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    col, row = start_point
    trans = dataset.GetGeoTransform()
    print(trans)
    print(row,col)
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return (px, py)


def gen_geoClips(imgPath, outPath, start_point, size=640):
    lc = gdal.Open(imgPath)
    im_width = lc.RasterXSize
    im_height = lc.RasterYSize
    im_geotrans = lc.GetGeoTransform()
    bandscount = lc.RasterCount
    im_proj = lc.GetProjection()

    start_point = imagexy2geo(lc, start_point)

    xValues = []
    yValues = []

    xValues.append(start_point[0])
    yValues.append(start_point[1])
    newform = []
    newform = list(im_geotrans)
    # print newform
    newform[0] = start_point[0] - im_geotrans[1] * int(size) / 2.0
    newform[3] = start_point[1] - im_geotrans[5] * int(size) / 2.0
    print(newform[0], newform[3])
    newformtuple = tuple(newform)

    x1 = start_point[0] - int(size) / 2 * im_geotrans[1]
    y1 = start_point[1] - int(size) / 2 * im_geotrans[5]
    x2 = start_point[0] + int(size) / 2 * im_geotrans[1]
    y2 = start_point[1] - int(size) / 2 * im_geotrans[5]
    x3 = start_point[0] - int(size) / 2 * im_geotrans[1]
    y3 = start_point[1] + int(size) / 2 * im_geotrans[5]
    x4 = start_point[0] + int(size) / 2 * im_geotrans[1]
    y4 = start_point[1] + int(size) / 2 * im_geotrans[5]
    Xpix = (x1 - im_geotrans[0]) / im_geotrans[1]
    # Xpix=(newform[0]-im_geotrans[0])

    Ypix = (newform[3] - im_geotrans[3]) / im_geotrans[5]

    pBuf = None
    pBuf = lc.ReadAsArray(int(Xpix), int(Ypix), int(size), int(size))
    # print pBuf.dtype.name
    driver = gdal.GetDriverByName("GTiff")
    create_option = []
    if 'int8' in pBuf.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in pBuf.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    outtif = outPath
    ds = driver.Create(outtif, int(size), int(size), int(bandscount), datatype, options=create_option)
    if ds == None:
        print("2222")
    ds.SetProjection(im_proj)
    ds.SetGeoTransform(newformtuple)
    ds.FlushCache()
    if bandscount > 1:
        for i in range(int(bandscount)):
            outBand = ds.GetRasterBand(i + 1)
            outBand.WriteArray(pBuf[i])
    else:
        outBand = ds.GetRasterBand(1)
        outBand.WriteArray(pBuf)
    ds.FlushCache()


def array_to_raster(array, dst_filename):
    """Array > Raster
    Save a raster from a C order array.
    :param array: ndarray
     """

    x_pixels, y_pixels, bandscount = array.shape

    datatype = gdal.GDT_Byte

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
           dst_filename,
           x_pixels,
           y_pixels,
           bandscount,
           datatype)

    if bandscount > 1:
        for i in range(int(bandscount)):
            dataset.GetRasterBand(i + 1).WriteArray(array[:, :, i])
    else:
        dataset.GetRasterBand(1).WriteArray(np.squeeze(array))
    dataset.FlushCache()

    return dataset, dataset.GetRasterBand(1)


if __name__ == '__main__':
    crop('imgRoot', 'outRoot', 576, 64, '.tif')
