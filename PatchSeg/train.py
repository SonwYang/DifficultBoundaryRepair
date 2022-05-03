from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from datasetGeo import BIPEDDataset
from loss import *
from config import Config
from cyclicLR import CyclicCosAnnealingLR, LearningRateWarmUP
from models.unetresnet import deeplabV3, UNetGlore, UNetResNet
from models.HRNet import HighResolutionNet
from models.net import FishUnet
import segmentation_models_pytorch as smp

import torchgeometry as tgm
import numpy as np
import time
import os
import cv2 as cv
import tqdm
import glob
from random import sample
from lookahead import Lookahead
import warnings
warnings.filterwarnings("ignore")


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


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = smp.UnetPlusPlus(in_channels=8, classes=3, encoder_weights=None).to(self.device)

        self.criterion_seg = WeightedFocalLoss2d()

        optimizer = torch.optim.AdamW([
                {'params': self.model.parameters()},
                # {'params': self.awl.parameters(), 'weight_decay': 0}
            ])
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

        train_dataset = BIPEDDataset(self.cfg.train_root)
        valid_dataset = BIPEDDataset(self.cfg.train_root)

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
        for batch_id, sample_batched in enumerate(dataloader):

            images = sample_batched['image'].to(self.device)  # BxCxHxW
            labels_seg = sample_batched['gt'].to(self.device)  # BxHxW
            file_name = sample_batched['file_name']

            segments = self.model(images)

            loss_seg = self.criterion_seg(segments, labels_seg)

            loss = loss_seg

            print(time.ctime(), 'validation, Epoch: {0} Sample {1}/{2} Loss: {3}' \
                  .format(epoch, batch_id, len(dataloader), loss.item()), end='\r')

            self.save_image_bacth_to_disk(segments, file_name)
            running_loss.append(loss.detach().item())
            return np.mean(np.array(running_loss))

    def save_image_bacth_to_disk(self, tensor, file_names):
        output_dir = self.cfg.valid_output_dir
        mkdir(output_dir)
        assert len(tensor.shape) == 4, tensor.shape
        for tensor_image, file_name in zip(tensor, file_names):
            image_vis = tgm.utils.tensor_to_image(torch.sigmoid(tensor_image))[..., 1]
            image_vis = (255.0 * (1.0 - image_vis)).astype(np.uint8)  #
            output_file_name = os.path.join(output_dir, f"{file_name}.png")
            cv.imwrite(output_file_name, image_vis)

    def train(self):
        train_loader, valid_loader = self.build_loader()
        best_loss = 1000000
        best_train_loss = 1000000
        valid_losses = []
        train_losses = []

        running_loss = []
        for epoch in range(1, self.cfg.num_epochs):
            self.model.train()
            for batch_id, sample_batched in enumerate(train_loader):

                images = sample_batched['image'].to(self.device)  # BxCxHxW
                labels_seg = sample_batched['gt'].to(self.device)  # BxHxW

                segments = self.model(images)

                loss_seg = self.criterion_seg(segments, labels_seg)

                loss = loss_seg

                self.optimizer.zero_grad()
                torch.autograd.backward([loss_seg])
                # loss.backward()
                self.optimizer.step()
                print(time.ctime(), 'training, Epoch: {0} Sample {1}/{2} Loss: {3}'\
                      .format(epoch, batch_id, len(train_loader), loss.item()), end='\r')
                running_loss.append(loss.detach().item())

            train_loss = np.mean(np.array(running_loss))

            valid_loss = self.validation(epoch, valid_loader)

            if epoch > 10:
                self.scheduler.after_scheduler.step(valid_loss)
            else:
                self.scheduler.step(epoch)

            lr = float(self.scheduler.after_scheduler.optimizer.param_groups[0]['lr'])

            if valid_loss < best_loss:
                torch.save(self.model, os.path.join(self.cfg.model_output, f'epoch{epoch}_model.pth'))
                print(f'find optimal model, loss {best_loss}==>{valid_loss} \n')
                best_loss = valid_loss

                # print(f'lr {lr:.8f} \n')
                valid_losses.append([valid_loss, lr])
                np.savetxt(os.path.join(self.cfg.model_output, 'valid_loss.txt'), valid_losses, fmt='%.6f')

            if train_loss < best_train_loss:
                torch.save(self.model, os.path.join(self.cfg.model_output, f'train_best_model.pth'))
                best_train_loss = train_loss
                train_losses.append([train_loss, lr])
                np.savetxt(os.path.join(self.cfg.model_output, 'train_loss.txt'), train_losses, fmt='%.6f')

            torch.save(self.model, os.path.join(self.cfg.model_output, f'last_model.pth'))

        # plt.ioff()
        # plt.show()

# python train.py --dataRoot ${your data path}
if __name__ == '__main__':
    import argparse
    from preprocessData import prepareData
    parser = argparse.ArgumentParser(
        description='''This is a code for training model.''')
    parser.add_argument('--dataRoot', type=str, default=r'D:\BaiduNetdiskDownload\GF3_Yangzhitang_Samples_Feature', help='path to the root of data')
    args = parser.parse_args()

    # print('The training dataset is preparing... Please wait!')
    # prepareData(args.dataRoot)
    config = Config()

    print("Everything is ok! It's time for training.")
    trainer = Trainer(config)
    trainer.train()





