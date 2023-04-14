#!/usr/bin/env python
# coding: utf-8

import random
import os
import re
import time
import argparse
from tqdm import tqdm
import datetime

import torch
from torch.utils.data import DataLoader
from models import detectors
from torch import nn

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(1)
writer = SummaryWriter('logs')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="../pytorch_ckpts/", type=str, dest='model_path', help='Pytorch model path.')
    parser.add_argument('--mode', default='train', type=str, dest='mode', help='AA:train AA models; Sm:train Sm models; Pg:train Pg models; train:train all models with same settings; test:test mode.')
    parser.add_argument('--gpu', type=str, dest='gpu_ids', help="GPU.")
    parser.add_argument('--log_path', default='../logs/pytorch', type=str, dest='log_path', help='Log folder.')

    # parser.add_argument('--gpu', type=str, dest='gpu_ids', help="GPU.")

    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    return args

def train(dataloader, model, loss_fn, optimizer, device):
    model = model.float()

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device).float()
        pred = model(X)
        y = y.unsqueeze(1)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            # if not batch == len(dataloader)-1:
            #     print(f"loss:{loss:>7f} [{current:>3d}/{size:>3d}]")
            # else:
            #     print(f"loss:{loss:>7f} [{size:>3d}/{size:>3d}]")
    return loss

def val(dataloader, model, loss_fn, device, best_acc=0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device).float()
            y = y.unsqueeze(1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += torch.round(pred).eq(y.view_as(pred)).sum().item()

    test_loss /= num_batches
    correct /= size

    if correct > best_acc:
        best_acc = correct
    
    print(f"Test Result: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:8f}, Current accuracy: {(100*best_acc):>0.1f}%", end=', ')
    print()

    return test_loss, correct

def main():
    args = parse_args()

    device = "cuda:"+args.gpu_ids if torch.cuda.is_available() else "cpu"
    print(f'Using {device} device')

    batch_size = 32
    # dropout_rate = 0.45
    decay_rate = 0.9
    lr = 1e-5

    # log folder
    # log_folder example: /data/zhaozhenghao/spectra_classification/log/train_10-20_13:15
    # save_folder example: train_10-20_13:15/
    # save_folder = args.mode+'_' + str(datetime.datetime.now())[5:-10]
    # save_folder = save_folder.replace(" ", "_")
    save_folder = args.mode+'_lr_'+str(lr)+'_batch_size_'+str(batch_size)
    log_folder = os.path.join(args.log_path, save_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # load data
    train_path = "../plots/10s_dataset_Pg/train"
    val_path = "../plots/10s_dataset_Pg/val"
    test_path = "../plots/10s_dataset_Pg/test"

    transform = ToTensor()
    
    train_data = ImageFolder(root=train_path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    val_data = ImageFolder(root=val_path, transform=transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    test_data = ImageFolder(root=test_path, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    if args.mode == 'train': # train model once
        # ckpt folder
        # model_folder example: /data/zhaozhenghao/spectra_classification/model/train_10-20_13:15
        model_folder = os.path.join(args.model_path, save_folder)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        model_path = os.path.join(model_folder, 'net.pth')

        model = detectors.image_model().to(device)

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

        epochs = 1000
        best_loss = 100
        last_loss = 100
        patience = 200
        trigger = 0
        for t in range(epochs):
            print(f"Epoch {t+1}/{epochs}", end=', ')
            loss = train(train_loader, model, loss_fn, optimizer, device)

            lr_scheduler.step()

            val_loss, val_acc = val(val_loader, model, loss_fn, device)

            writer.add_scalar('Training Loss', loss, t+1)
            writer.add_scalar('Validation Loss', val_loss, t+1)
            writer.add_scalar('Validation Accuracy', val_acc, t+1)

            # overwrite best model for current parameter settings
            if val_loss < best_loss:
                # print(f'{val_loss} < {best_loss}, saving ckpt...')
                best_loss = val_loss
                torch.save(model.state_dict(), model_path)

            # early stopping trigger
            if val_loss > last_loss:
                trigger += 1
            if trigger >= patience:
                break
            
        writer.close()

        best_state_dict = torch.load(model_path)
        eval_model = detectors.image_model().to(device)
        eval_model.load_state_dict(best_state_dict)
        eval_model.eval()
        val_loss, val_acc = val(val_loader, eval_model, loss_fn, device)
        print()
        test_loss, test_acc = val(test_loader, eval_model, loss_fn, device)
        print()

if __name__ == '__main__':
    main()