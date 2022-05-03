import gdalTools
import os
import cv2
import numpy as np
from skimage.morphology import square, dilation

if __name__ == '__main__':
    imgPath = r"D:\MyWorkSpace\paper\fishpond\fishpondCode\fishpondPredict\predict_ours_model+pointrend\SOTA.tif"
    im_proj, im_geotrans, im_width, im_height, im_data = gdalTools.read_img(imgPath)
    im_data = np.where(im_data == 1, 1, 0)
    im_data = dilation(im_data, square(2))
    im_data = np.where(im_data > 0, 1, 0)

    outImgPath = imgPath.replace("SOTA.tif", "final.tif")
    gdalTools.write_img(outImgPath, im_proj, im_geotrans, im_data)