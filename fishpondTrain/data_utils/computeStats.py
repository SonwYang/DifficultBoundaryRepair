import numpy as np
import gdalTools
import glob


def computeMS(ImgRoot):
    imglist = glob.glob(f'{ImgRoot}/*.png')
    means = []
    stds = []
    for imgPath in imglist:
        im_proj, im_geotrans, im_width, im_height, im_data = gdalTools.read_img(imgPath)
        im_data = im_data / 255.
        means.append(im_data.mean(axis=(1, 2)))
        stds.append(im_data.std(axis=(1, 2)))
    return np.array(means).mean(axis=0), np.array(stds).mean(axis=0)


if __name__ == '__main__':
    mean, std = computeMS(r'D:\MyWorkSpace\MyUtilsCode\CroplandAI\CroplandTrain\data\train_images')
    print(mean, std)

