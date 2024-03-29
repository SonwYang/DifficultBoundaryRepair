# Introduction

This repository is the official implementation of [Coastal Aquaculture Extraction Using GF-3 Fully Polarimetric SAR Imagery: A Framework Integrating UNet++ with Marker-Controlled Watershed Segmentation](https://www.mdpi.com/2072-4292/15/9/2246). The main work flow is as follows:

![image](1.png)

This method can be split into two steps: 1. getting coarse segmentation via U-Net; 2. According to the difficult boundary proposals, repairing the boundary in a patch.

![image](2.png)

# Requirements

- torch
- torchvision
- opencv-python
- gdal

# How to use

1. train your custom coarse segmentation model

   ```
   cd fishpondTrain
   python train.py --dataRoot ${your data root} --in_chs 1 --num_classes 2
   ```

2. prediction(coarse segmentation)

   ```
   cd fishpondPredict
   python predict.py --data_path ${the path of image} --seg_model_path ${coarse segmentation model}
   ```

3. patch repair (training and prediction)


   ```
   cd PatchSeg
   python train.py --dataRoot ${your data path}
   python PatchPredict.py --imgPath entropy_shannon_subset_Feature.tif --CoarseSegPath coarse.tif --modelPath patch.pth --outPath final.tif
   ```

# Final result

![image](3.png)

# Acknowledgement

Thanks for completing this repo with Dr. Yu's kind help.

