from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, watershed, opening
from pytorch_toolbelt.inference.tta import TTAWrapper, fliplr_image2mask, d4_image2mask
from skimage.morphology import opening, closing, square
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
    coarseImgPath = r'D:\MyWorkSpace\paper\fishpond\data_evaluation\predict_unetplus3+pf\poly.tif'
    FusionPath = r'D:\MyWorkSpace\paper\fishpond\data_evaluation\predict_unetplus3+pf\fusion.tif'
    im_proj, im_geotrans, im_width, im_height, im_data = gdalTools.read_img(coarseImgPath)
    mask1 = np.where(im_data > 0, 1, 0)
    mask2 = np.where(im_data == 1, 1, 0)
    result = my_watershed(mask1, mask2)
    result[result > 0] = 1
    # result = bool(result)
    result = remove_small_objects(result, 64)
    result = result.astype(np.uint8)
    gdalTools.write_img(FusionPath, im_proj, im_geotrans, result)