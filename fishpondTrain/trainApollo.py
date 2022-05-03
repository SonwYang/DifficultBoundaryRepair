import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from datasetGeo import BIPEDDataset
from loss import *
from config import Config
from cyclicLR import CyclicCosAnnealingLR, LearningRateWarmUP
from segmentation_models_pytorch import Unet
from models.EESNet import *
from models.SOED import SOED, SOED2_act
import torchgeometry as tgm
import numpy as np
import time

import cv2 as cv
import tqdm
import glob
from random import sample
from lookahead import Lookahead
import matplotlib.pyplot as plt
from skimage.morphology import square, dilation
from skimage import morphology
from preprocessData import prepareData
from data_utils.computeStats import computeMS
from utils.optimers.apollo import Apollo
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img


def weight_init(m):
    if isinstance(m, (nn.Conv2d, )):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0,)
        if m.weight.data.shape == torch.Size([1,6,1,1]):
            torch.nn.init.constant_(m.weight,0.2)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):

        torch.nn.init.normal_(m.weight,mean=0, std=0.01)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def compute_centerline_metrics(gt, pred, buffer=7):
    gt = np.where(gt > 0, 1, 0)
    pred = np.where(pred > 0, 1, 0)
    length_gt = np.sum(gt)
    length_pred = np.sum(pred)
    pred_mask = dilation(pred.copy(), square(buffer))
    pred_pos = np.where((pred_mask + gt) == 2, 1, 0)
    length_pred_pos = np.sum(pred_pos)

    gt_mask = dilation(gt.copy(), square(buffer))
    gt_pos = np.where((gt_mask + pred) == 2, 1, 0)
    length_gt_pos = np.sum(gt_pos)

    quality = length_pred_pos / (length_pred + length_gt - length_gt_pos)

    return quality


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.edge_weight = [0.5, 0.75, 0.75, 0.5, 1.1]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = EESNetV3(3).to(self.device).apply(weight_init)
        # if torch.cuda.device_count() > 1:
        #     print("using multi gpu")
        #     self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        # else:
        #     print('using one gpu')
        self.awl = AutomaticWeightedLoss(num=2)
        self.criterion_seg = WeightedFocalLoss2d()
        self.criterion_edge = bdcn_lossV3
        self.criterion_ori = CrossEntropyLoss2d()
        self.criterion_point = FocalLoss()
        # optimizer = torch.optim.AdamW([
        #         {'params': self.model.parameters()},
        #         # {'params': self.awl.parameters(), 'weight_decay': 0}
        #     ])
        optimizer = Apollo(self.model.parameters(), 0.0005)
        self.optimizer = Lookahead(optimizer)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, verbose=True)
        self.scheduler = LearningRateWarmUP(optimizer=optimizer, target_iteration=10, target_lr=0.0005,
                                            after_scheduler=scheduler)
        mkdir(cfg.model_output)

    def load_net(self, resume):
        self.model = torch.load(resume,  map_location=self.device)
        print('load pre-trained model successfully')

    def build_loader(self):
        imglist = glob.glob(f'{self.cfg.train_root}/*')
        indices = list(range(len(imglist)))
        indices = sample(indices, len(indices))
        split = int(np.floor(0.15 * len(imglist)))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        print(f'Total images {len(imglist)}')
        print(f'No of train images {len(train_idx)}')
        print(f'No of validation images {len(valid_idx)}')

        train_dataset = BIPEDDataset(self.cfg.train_root,
                                     crop_size=self.cfg.img_width,
                                     mean=self.cfg.mean,
                                     std=self.cfg.std)
        valid_dataset = BIPEDDataset(self.cfg.train_root,
                                     crop_size=self.cfg.img_width,
                                     mean=self.cfg.mean,
                                     std=self.cfg.std)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.cfg.batch_size,
                                  num_workers=self.cfg.num_workers,
                                  shuffle=False,
                                  sampler=train_sampler,
                                  drop_last=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.cfg.batch_size,
                                  num_workers=self.cfg.num_workers,
                                  shuffle=False,
                                  sampler=valid_sampler,
                                  drop_last=True)
        return train_loader, valid_loader

    def validation(self, epoch, dataloader):
        self.model.eval()
        running_loss = []
        qualities = []

        for batch_id, sample_batched in enumerate(dataloader):

            images = sample_batched['image'].to(self.device)  # BxCxHxW
            labels_seg = sample_batched['gt'].to(self.device)  # BxHxW
            labels_center = sample_batched['center'].to(self.device)
            file_name = sample_batched['file_name']
            labels_gaussian = sample_batched['gaussian'].to(self.device)

            segments, centers = self.model(images)

            loss_seg = self.criterion_seg(segments, labels_seg)
            # loss_boundary = sum([self.criterion_edge(boundary, labels_boundary, l_weight=l_w) for boundary, l_w
            #                      in zip(boundarys, self.edge_weight)]) / len(boundarys)
            loss_center = sum([self.criterion_edge(center, labels_center, l_weight=l_w) for center, l_w
                               in zip(centers, self.edge_weight)]) / len(centers)

            loss = loss_center + loss_seg

            print(time.ctime(), 'validation, Epoch: {0} Sample {1}/{2} Loss: {3}' \
                  .format(epoch, batch_id, len(dataloader), loss.item()), end='\r')

            quality = self.save_image_bacth_to_disk(centers[-1], labels_center, file_name)
            qualities.append(quality)
            running_loss.append(loss.detach().item())
            return np.mean(np.array(running_loss)), np.mean(qualities)

    def save_image_bacth_to_disk(self, tensor,  labels_center, file_names):
        b, c, h, w = tensor.size()
        output_dir = self.cfg.valid_output_dir
        mkdir(output_dir)
        assert len(tensor.shape) == 4, tensor.shape

        quality_list = []
        for tensor_image, label_center, file_name in zip(tensor, labels_center, file_names):
            label_center = label_center.cpu().squeeze().numpy()
            image_vis = tgm.utils.tensor_to_image(torch.sigmoid(tensor_image))[..., 0]
            image_vis = (255.0 * image_vis).astype(np.uint8)  #
            # image_vis = np.stack((image_vis,) * 3).transpose(1, 2, 0)

            ret2, th2 = cv.threshold(image_vis, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            th2 = np.where(th2 > 0, 1, 0)
            skeleton = morphology.skeletonize(th2)
            skeleton = np.where(skeleton > 0, 1, 0).astype(np.uint8)
            quality = compute_centerline_metrics(label_center, skeleton)
            quality_list.append(quality)
            output_file_name = os.path.join(output_dir, f"{file_name}.png")
            cv.imwrite(output_file_name, image_vis)
        return np.mean(quality_list)

    def train(self):
        train_loader, valid_loader = self.build_loader()
        best_loss = 1000000
        best_train_loss = 1000000
        valid_losses = []
        train_losses = []

        running_loss = []
        plt.ion()
        x = list(range(1, self.cfg.num_epochs))  # epoch array
        for epoch in range(1, self.cfg.num_epochs):
            self.model.train()
            for batch_id, sample_batched in enumerate(train_loader):
                # if batch_id > 2:
                #     break

                images = sample_batched['image'].to(self.device)  # BxCxHxW
                labels_seg = sample_batched['gt'].to(self.device)  # BxHxW
                labels_center = sample_batched['center'].to(self.device)

                segments, centers = self.model(images)

                loss_seg = self.criterion_seg(segments, labels_seg)

                loss_center = sum([self.criterion_edge(center, labels_center, l_weight=l_w) for center, l_w
                                 in zip(centers, self.edge_weight)]) / len(centers)

                loss = self.awl([loss_seg, loss_center])

                self.optimizer.zero_grad()
                # torch.autograd.backward([loss_seg, loss_center])
                loss.backward()
                self.optimizer.step()

                print(time.ctime(), 'training, Epoch: {0} Sample {1}/{2} Loss: {3}'\
                      .format(epoch, batch_id, len(train_loader), loss.item()), end='\r')
                running_loss.append(loss.detach().item())

            train_loss = np.mean(np.array(running_loss))

            valid_loss, quality = self.validation(epoch, valid_loader)

            if epoch > 10:
                self.scheduler.after_scheduler.step(valid_loss)
            else:
                self.scheduler.step(epoch)

            lr = float(self.scheduler.after_scheduler.optimizer.param_groups[0]['lr'])

            if valid_loss < best_loss:
                torch.save(self.model, os.path.join(self.cfg.model_output, f'epoch{epoch}_model.pth'))
                torch.save(self.model, self.cfg.bestModelPath)
                print(f'find optimal model, loss {best_loss}==>{valid_loss} \n')
                best_loss = valid_loss

                # print(f'lr {lr:.8f} \n')
                valid_losses.append([valid_loss, lr])
                np.savetxt(os.path.join(self.cfg.model_output, 'valid_loss.txt'), valid_losses, fmt='%.6f')

            # if valid_loss < best_loss:
            #     print(f'find optimal model, loss {best_loss:6f}==>{valid_loss:6f}\n')
            #     torch.save(self.model, os.path.join(self.cfg.model_output, f'train_best_loss_model.pth'))
            #     best_loss = valid_loss
            #     train_losses.append([train_loss, lr])
            #     np.savetxt(os.path.join(self.cfg.model_output, 'train_loss.txt'), train_losses, fmt='%.6f')

            torch.save(self.model, os.path.join(self.cfg.model_output, f'last_model.pth'))

        # plt.ioff()
        # plt.show()


if __name__ == '__main__':
    config = Config()

    f = open('configTrain.txt')
    data = f.readlines()
    data_path = data[0].replace('\n', '')
    config.bestModelPath = data[1].replace('\n', '')
    prepareData(data_path)
    os.system('python Redundancy_cropUtil.py')
    mean, std = computeMS(config.train_root)
    config.mean = mean
    config.std = std
    print(f"This dataset, mean:{mean}, std:{std}")

    trainer = Trainer(config)
    # trainer.load_net("D:\MyWorkSpace\MyUtilsCode\segmentation\CroplandPredict\model_reults\EESNet5_wm.pth")
    trainer.train()



