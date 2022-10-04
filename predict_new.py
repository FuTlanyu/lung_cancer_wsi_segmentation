import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
import util
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from loss import FocalLoss, DiceLoss, FocalLossPlusDiceLoss
from dice_loss import dice_coeff

if __name__ == '__main__':

    # args
    load = util.model_dir + 'level0_40k/CP_epoch10_dice0.7727556594333288.pth'
    img_scale = 0.5
    batch_size = 1
    dir_img = 'E:/UnetSegPy-Data/data/patch/test/max/patch_image/89/'
    dir_mask = 'E:/UnetSegPy-Data/data/patch/test/max/gt_mask/89/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=3, n_classes=1)

    if load:
        net.load_state_dict(
            torch.load(load, map_location=device)
        )
        print(f'Model loaded from {load}')

    net.to(device=device)

    test_dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_test = len(test_dataset)
    print(n_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    net.eval()
    tot = 0

    for batch in test_loader:
        imgs = batch['image']
        true_masks = batch['mask']

        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_type = torch.float32 if net.n_classes == 1 else torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)

        mask_pred = net(imgs)

        for true_mask, pred in zip(true_masks, mask_pred):
            pred = (pred > 0.5).float()
            if net.n_classes > 1:
                tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
            else:
                tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()

    test_score = tot / n_test

    if net.n_classes > 1:
        logging.info('Validation cross entropy: {}'.format(val_score))

    else:
        print('Validation Dice Coeff: {}'.format(test_score))
