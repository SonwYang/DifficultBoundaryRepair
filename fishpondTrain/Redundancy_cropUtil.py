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


def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def subImg(img, i, j, targetSize, PaddingSize, height, width):
    if (i + 1) * targetSize < height and (j + 1) * targetSize < width:
        temp_img = img[targetSize * i: targetSize * i + targetSize + PaddingSize,
                   targetSize * j: targetSize * j + targetSize + PaddingSize]
    elif (i + 1) * targetSize < height and (j + 1) * targetSize > width:
        temp_img = img[targetSize * i: targetSize * i + targetSize + PaddingSize,
                   width - targetSize - PaddingSize: width]
    elif (i + 1) * targetSize > height and (j + 1) * targetSize < width:
        temp_img = img[height - targetSize - PaddingSize: height,
                   targetSize * j: targetSize * j + targetSize + PaddingSize]
    else:
        temp_img = img[height - targetSize - PaddingSize: height, width - targetSize - PaddingSize: width]
    return temp_img


def readImg(imgPath):
    data = gdal.Open(imgPath)
    lastChannel = data.RasterCount + 1
    arr = [data.GetRasterBand(idx).ReadAsArray() for idx in range(1, lastChannel)]
    return np.dstack(arr)


def crop(img_dir, label_dir, targetSize, PaddingSize, ImgSuffix, LabelSuffix):
    labels_list = glob.glob("./plough/train_labels/*.tif")

    imgs_num = len(labels_list)
    print("imgs_num:{}".format(labels_list))
    make_dir(img_dir)
    make_dir(label_dir)

    for k in tqdm.tqdm(range(imgs_num)):
        label = cv.imread(labels_list[k])
        imgName = os.path.split(labels_list[k])[-1].split('.')[0]

        imgPath = labels_list[k].replace('labels', 'images').replace('_LINE.tif', '.tif')
        img = readImg(imgPath)

        height, width = img.shape[0], img.shape[1]
        rows, cols = height // targetSize + 1, width // targetSize + 1
        subImg_num = 0
        for i in range(rows):
            for j in range(cols):
                temp_img = subImg(img, i, j, targetSize, PaddingSize, height, width)
                temp_label = subImg(label, i, j, targetSize, PaddingSize, height, width)
                if args.isImg:
                    tempName = imgName + "_" + str(subImg_num) + ImgSuffix
                    labelName = imgName + "_" + str(subImg_num) + LabelSuffix

                    if ImgSuffix == '.npy':
                        np.save(img_dir + '/' + tempName, temp_img.astype(np.uint8))
                    else:
                        io.imsave(img_dir + '/' + tempName, temp_img.astype(np.uint8))
                    io.imsave(label_dir + '/' + labelName, temp_label)

                subImg_num += 1


def crop2(img_dir, targetSize, PaddingSize, ImgSuffix):
    labels_list = glob.glob("./plough/temps3/*.tif")

    imgs_num = len(labels_list)
    print("imgs_num:{}".format(labels_list))
    make_dir(img_dir)

    for k in tqdm.tqdm(range(imgs_num)):
        label = cv.imread(labels_list[k])
        imgName = os.path.split(labels_list[k])[-1].split('.')[0]

        height, width = label.shape[0], label.shape[1]
        rows, cols = height // targetSize + 1, width // targetSize + 1
        subImg_num = 0
        for i in range(rows):
            for j in range(cols):
                temp_label = subImg(label, i, j, targetSize, PaddingSize, height, width)
                if args.isImg:
                    tempName = imgName + "_" + str(subImg_num) + ImgSuffix

                    if ImgSuffix == '.npy':
                        np.save(img_dir + '/' + tempName, temp_label.astype(np.uint8))
                    else:
                        io.imsave(img_dir + '/' + tempName, temp_label.astype(np.uint8))

                subImg_num += 1

def crop3(imgRoot, img_dir, targetSize, PaddingSize, ImgSuffix):
    labels_list = glob.glob(f"{imgRoot}/*.tif")

    imgs_num = len(labels_list)
    print("imgs_num:{}".format(labels_list))
    make_dir(img_dir)

    for k in tqdm.tqdm(range(imgs_num)):
        label = io.imread(labels_list[k])
        imgName = os.path.split(labels_list[k])[-1].split('.')[0]

        height, width = label.shape[0], label.shape[1]
        rows, cols = height // targetSize + 1, width // targetSize + 1
        subImg_num = 0
        for i in range(rows):
            for j in range(cols):
                temp_label = subImg(label, i, j, targetSize, PaddingSize, height, width)
                if args.isImg:
                    tempName = imgName + "_" + str(subImg_num) + ImgSuffix

                    if ImgSuffix == '.npy':
                        np.save(img_dir + '/' + tempName, temp_label.astype(np.uint8))
                    else:
                        io.imsave(img_dir + '/' + tempName, temp_label.astype(np.uint8))

                subImg_num += 1


if __name__ == '__main__':
    import gdalTools
    import shutil
    parse = argparse.ArgumentParser()
    # parse.add_argument("--root", type=str, default="./val", help='the path of input')
    parse.add_argument("--img_dir", type=str, default="./train_images", help='the path of images output')
    parse.add_argument("--label_dir", type=str, default="./train_labels", help='the path of labels output')
    parse.add_argument("--targetSize", type=int, default=512, help='the size of target')
    parse.add_argument("--PaddingSize", type=int, default=0, help='the size of padding')
    parse.add_argument("--LabelSuffix", type=str, default=".png", help='the suffix of label')
    parse.add_argument("--ImgSuffix", type=str, default=".png", help='the suffix of image')
    parse.add_argument("--isImg", type=bool, default=True, help='Img is true, np is false')

    args = parse.parse_args()
    # crop(args.img_dir, args.label_dir, args.targetSize, args.PaddingSize, args.ImgSuffix, args.LabelSuffix)

    rootList = ['gt']
    outRoot = 'data'
    if os.path.exists(outRoot):
        shutil.rmtree(outRoot)

    gdalTools.mkdir(outRoot)
    for root in rootList:
        subRoot = os.path.join(outRoot, root)

        crop3(root, subRoot, args.targetSize, args.PaddingSize, args.ImgSuffix)

