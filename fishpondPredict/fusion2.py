from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, watershed, opening
from pytorch_toolbelt.inference.tta import TTAWrapper, fliplr_image2mask, d4_image2mask
from skimage.morphology import opening, closing, square, erosion, dilation
import gdalTools
import numpy as np
from photutils import find_peaks


###mask1   big    mask2 small
def my_watershed(mask1, mask2):
    """
    watershed from mask1 with markers from mask2
    """
    markers = ndi.label(mask2, output=np.uint32)[0]
    labels = watershed(mask1, markers, mask=mask1, watershed_line=True)
    return labels


if __name__ == '__main__':
    coarseImgPath = r'D:\MyWorkSpace\paper\fishpond\data_evaluation\predict_unetPlus2+pf\poly.tif'
    markerPath = r'D:\MyWorkSpace\paper\fishpond\data_evaluation\predict_unetplus3\poly.tif'
    FusionPath = r'D:\MyWorkSpace\paper\fishpond\data_evaluation\predict_unetPlus2+pf\fusion2.tif'
    im_proj, im_geotrans, im_width, im_height, im_data = gdalTools.read_img(coarseImgPath)
    im_proj, im_geotrans, im_width, im_height, marker = gdalTools.read_img(markerPath)
    mask1 = np.where(im_data > 0, 1, 0)
    mask2 = np.where(marker == 1, 1, 0) + mask1
    mask2 = np.where(mask2 == 2, 1, 0)
    result = my_watershed(mask1, mask2)
    result[result > 0] = 1
    # waterLine = mask1 - result
    # waterLine = np.where(waterLine > 0, 1, 0)
    # waterLine = dilation(waterLine, square(2))
    # result = bool(result)
    result = remove_small_objects(result, 100)
    result = result.astype(np.uint8)
    #result = erosion(result, square(2))
    gdalTools.write_img(FusionPath, im_proj, im_geotrans, result)