#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import os
import re
import pickle
import sys
import time
import argparse
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from utils import datasets
from models import traditional_methods
from models import classifiers

normalized_datapath = '/home/exx/Spoon/spectra_classification/data/normalized_data.txt'

log_folder = '/home/exx/Spoon/spectra_classification/ready/logs/'
log_path = os.path.join(log_folder,'data_augmentation.txt')
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="/home/exx/Spoon/spectra_classification/ready/model/", type=str, dest='model_path', help='Model path.')
    parser.add_argument('--use_aug', default='True', type=str, dest='use_aug', help='Use data augmentation, True or False.')
    parser.add_argument('--expand_dim', default='True', type=str, dest='expand_dim', help='Expand data dimension, True or False. MUST be set to True if use data augmentation.')
    parser.add_argument('--mode', default='all', type=str, dest='mode', help='AA:train AA models; Sm:train Sm models; Pg:train Pg models; All:train all models with same settings; test:test mode.')
    parser.add_argument('--data', default='30s', type=str, dest='data', help='Select from 30s, 10s, all')

    args = parser.parse_args()

    if args.use_aug == 'True':
        args.use_aug = True
    elif args.use_aug == 'False':
        args.use_aug = False

    if args.expand_dim == 'True':
        args.expand_dim = True
    elif args.expand_dim == 'False':
        args.expand_dim = False

    return args

def writelog(instring, filepath):
    with open(filepath,'a') as f:
        f.write(instring+'\n')
    print(instring)

def conclude_results(AA_results, Sm_results, Pg_results, test_labels):
    results = []
    results.append(list(AA_results))
    results.append(list(Sm_results))
    results.append(list(Pg_results))
    results = np.array(results)

    results = np.reshape(results,(3,results.shape[1]))
    results = results.transpose()

    total_acc = accuracy_score(results, test_labels)
    print(f'{total_acc=}')
    print(classification_report(test_labels, results))

    return total_acc

def train(train_data, test_data, test_labels, AA_train_labels, AA_test_labels, Sm_train_labels, Sm_test_labels, Pg_train_labels, Pg_test_labels, data_aug=[], n_dim=35, Conv=False):
    writelog('Using data_augmentation methods:'+str(data_aug), log_path)
    AA_results, AA_acc, AA_loss, AA_model = classifiers.train_AA(train_data, AA_train_labels, test_data, AA_test_labels, data_augmentation_f=data_aug, n_dim=n_dim)
    Sm_results, Sm_acc, Sm_loss, Sm_model = classifiers.train_Sm(train_data, Sm_train_labels, test_data, Sm_test_labels, data_augmentation_f=data_aug, n_dim=n_dim)
    Pg_results, Pg_acc, Pg_loss, Pg_model = classifiers.train_Pg(train_data, Pg_train_labels, test_data, Pg_test_labels, data_augmentation_f=data_aug, n_dim=n_dim)

    total_acc = conclude_results(AA_results, Sm_results, Pg_results, test_labels)
    writelog('Total accuracy: '+str(total_acc), log_path)
    writelog('', log_path)

def main():
    args = parse_args()

    expand_dim = args.expand_dim
    if args.use_aug == True:
        expand_dim = True
        data_aug_list = ['jitter', 'permutation', 'spawner', 'wdba', 'rgw', 'dgw']
    else:
        data_aug_list = []

    # expand_dim and data_augmentation_f should be set to True together
    train_data, test_data, test_labels, AA_train_labels, AA_test_labels, Sm_train_labels, Sm_test_labels, Pg_train_labels, Pg_test_labels = datasets.data_preprocess(n_dim=35, expand_dim=expand_dim, dataset=args.data)
    if args.mode == 'all':
        train(train_data, test_data, test_labels, AA_train_labels, AA_test_labels, Sm_train_labels, Sm_test_labels, Pg_train_labels, Pg_test_labels, data_aug=data_aug_list)

    elif args.mode in ['AA','Sm','Pg']:
        acc_list = []
        loss_list = []
        for dropout in tqdm(np.arange(0.1, 0.9, 0.05)):
            for lr in [1e-2,1e-3,1e-4,1e-5]:
                for hidden_size in np.arange(40,200,20):
                    if args.mode == 'AA':
                        results, acc, loss, model = classifiers.train_AA(train_data, AA_train_labels, test_data, AA_test_labels, data_augmentation_f=data_aug_list, n_dim=35, dropout=dropout, lr=lr, hidden_size=hidden_size)
                    elif args.mode == 'Sm':
                        results, acc, loss, model = classifiers.train_Sm(train_data, Sm_train_labels, test_data, Sm_test_labels, data_augmentation_f=data_aug_list, n_dim=35, dropout=dropout, lr=lr, hidden_size=hidden_size)
                    elif args.mode == 'Pg':
                        results, acc, loss, model = classifiers.train_Pg(train_data, Pg_train_labels, test_data, Pg_test_labels, data_augmentation_f=data_aug_list, n_dim=35, dropout=dropout, lr=lr, hidden_size=hidden_size)
                    else:
                        exit()
                    acc_list.append(acc)
                    loss_list.append(loss)
                    if acc >= max(acc_list) or loss <= min(loss_list):
                        flagTF = lambda x : 'T' if (x==True) else 'F'
                        dataAug = flagTF(args.use_aug)
                        expandDim = flagTF(args.expand_dim)
                        model_folder = os.path.join(args.model_path, args.mode, 'dataAug_'+dataAug+'_expandDim_'+expandDim+'_dataset_'+args.data)
                        if not os.path.exists(model_folder):
                            os.makedirs(model_folder)
                        model.save(os.path.join(model_folder, 'dropout_'+str(round(dropout,2))+'_lr_'+str(lr)+'_hidden_size_'+str(hidden_size)+'_acc_'+str(round(acc,3))+'_loss_'+str(round(loss,3))+'.h5'))

    elif args.mode == 'test':
        AA_results, AA_acc, AA_loss, AA_model = classifiers.train_AA(train_data, AA_train_labels, test_data, AA_test_labels, ckpt_path='../model/AA/dataAug_F_expandDim_F/dropout_0.15_lr_1e-05_hidden_size_40_acc_0.971_loss_0.117.h5', data_augmentation_f=data_aug_list, n_dim=35, mode='test', hidden_size=40)
        Sm_results, Sm_acc, Sm_loss, Sm_model = classifiers.train_Sm(train_data, Sm_train_labels, test_data, Sm_test_labels, ckpt_path='../model/Sm/dataAug_F_expandDim_F/dropout_0.25_lr_1e-05_hidden_size_40_acc_0.934_loss_0.289.h5', data_augmentation_f=data_aug_list, n_dim=35, mode='test', hidden_size=40)
        Pg_results, Pg_acc, Pg_loss, Pg_model = classifiers.train_Pg(train_data_expandDim, Pg_train_labels, test_data, Pg_test_labels, ckpt_path='../model/Pg/dataAug_T_expandDim_T/dropout_0.3_lr_1e-05_hidden_size_80_acc_0.971_loss_0.132.h5', data_augmentation_f=data_aug_list, n_dim=35, mode='test', hidden_size=80)

        total_acc = conclude_results(AA_results, Sm_results, Pg_results, test_labels)
        print(f'{AA_acc=}')
        print(f'{Sm_acc=}')
        print(f'{Pg_acc=}')
        writelog('Total accuracy: '+str(total_acc), log_path)
        writelog('', log_path)
    else:
        print('unknown mode... EXIT!')
        exit()

if __name__ == '__main__':
    main()
