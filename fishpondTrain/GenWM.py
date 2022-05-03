import numpy as np
from scipy import ndimage
from osgeo import gdal
import gdalTools

# class balance weight map
def balancewm(mask):
    wc = np.empty(mask.shape)
    classes = np.unique(mask)
    freq = [ 1.0 / np.sum(mask==i) for i in classes ]
    freq /= max(freq)

    for i in range(len(classes)):
        wc[mask == classes[i]] = freq[i]

    return wc

# unet weight map
def unetwm(mask, w0=5, sigma=50):
    mask = mask.astype('float')
    wc = balancewm(mask)

    cells, cellscount = ndimage.measurements.label(mask == 1)
    # maps = np.zeros((mask.shape[0],mask.shape[1],cellscount))
    d1 = np.ones_like(mask) * np.Infinity
    d2 = np.ones_like(mask) * np.Infinity
    for ci in range(1, cellscount + 1):
        dstranf = ndimage.distance_transform_edt(cells != ci)
        d1 = np.amin(np.concatenate((dstranf[:, :, np.newaxis], d1[:, :, np.newaxis]), axis=2), axis=2)
        ind = np.argmin(np.concatenate((dstranf[:, :, np.newaxis], d1[:, :, np.newaxis]), axis=2), axis=2)
        dstranf[ind == 0] = np.Infinity
        if cellscount > 1:
            d2 = np.amin(np.concatenate((dstranf[:, :, np.newaxis], d2[:, :, np.newaxis]), axis=2), axis=2)
        else:
            d2 = d1.copy()

    uwm = 1 + wc + (mask == 0).astype('float') * w0 * np.exp((-(d1 + d2) ** 2) / (2 * sigma)).astype('float')

    return uwm

def distranfwm(mask, beta=3):
    mask = mask.astype('float')
    wc = balancewm(mask)

    dwm = ndimage.distance_transform_edt(mask != 1)
    dwm[dwm > beta] = beta
    dwm = wc + (1.0 - dwm / beta) + 1

    return dwm

def normal_img(img):
    # low_thresold = np.percentile(img, 5)
    # high_thresold = np.percentile(img, 95)
    # img = np.clip(img, low_thresold, high_thresold)
    min = np.min(img)
    max = np.max(img)
    img = (img - min) / (max - min) * 255
    return img.astype(np.uint8)


if __name__ == '__main__':
    import os
    import glob
    import matplotlib.pyplot as plt
    imgList = glob.glob("./data/train_labels/*.tif")
    outRoot = './data/train_wms'
    gdalTools.mkdir(outRoot)
    for imgPath in imgList:
        im_proj, im_geotrans, im_width, im_height, im_data = gdalTools.read_img(imgPath)
        wm = distranfwm(im_data)
        wm = normal_img(wm)
        baseName = os.path.basename(imgPath)
        outPath = os.path.join(outRoot, baseName)

        # cmap = 'nipy_spectral'
        # plt.imshow(wm, cmap=plt.get_cmap(cmap))
        # plt.colorbar()
        # plt.show()

        gdalTools.write_img(outPath, im_proj, im_geotrans, wm)
