import cv2 as cv
import gdalTools
import os
import numpy as np

if __name__ == '__main__':
    imgPath = r'D:\MyWorkSpace\paper\fishpond\fishpond_prediction\test\test.tif'
    outPath = r'D:\MyWorkSpace\paper\fishpond\fishpond_prediction\test\test_binary.tif'
    im_proj, im_geotrans, im_width, im_height, im_data = gdalTools.read_img(imgPath)
    im_data = np.array(im_data[0])
    print(im_data.shape)
    ret2, th2 = cv.threshold(im_data, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    th2 = np.where(th2 > 0, 1, 0)
    gdalTools.write_img(outPath, im_proj, im_geotrans, th2)