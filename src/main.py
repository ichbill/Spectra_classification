#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import os
import re
import pickle
import sys
import time

from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from tensorflow import keras

from utils import datasets
from utils import plot
from models import traditional_methods
from models import classifiers

model_path = "/home/exx/Spoon/spectra_classification/model/"

normalized_datapath = '/home/exx/Spoon/spectra_classification/data/normalized_data.txt'

log_path = '/home/exx/Spoon/spectra_classification/logs/data_augmentation.txt'

# species_dict = {0:'AA', 1:'Fn', 2:'Pg'}

data_aug_list = ['jitter', 'permutation', 'spawener', 'wdba', 'rgw', 'dgw']

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
    print("results.shape:", results.shape)
    # print(results)
    results = np.reshape(results,(3,137))
    results = results.transpose()
    print(f'{results.shape=}')

    total_acc = accuracy_score(results, test_labels)
    print(f'{total_acc=}')
    print(classification_report(test_labels, results))
    return total_acc

def train(train_data, test_data, test_labels, AA_train_labels, AA_test_labels, Sm_train_labels, Sm_test_labels, Pg_train_labels, Pg_test_labels, data_aug=[], n_dim=35, Conv=False):
    writelog('Using data_augmentation methods:'+str(data_aug), log_path)
    AA_results = classifiers.train_AA(train_data, AA_train_labels, test_data, AA_test_labels, data_augmentation_f=data_aug, n_dim=n_dim)
    Sm_results = classifiers.train_Sm(train_data, Sm_train_labels, test_data, Sm_test_labels, data_augmentation_f=data_aug, n_dim=n_dim)
    Pg_results = classifiers.train_Pg(train_data, Pg_train_labels, test_data, Pg_test_labels, data_augmentation_f=data_aug, n_dim=n_dim)

    total_acc = conclude_results(AA_results, Sm_results, Pg_results, test_labels)
    writelog('Total accuracy: '+str(total_acc), log_path)
    writelog('', log_path)

if __name__ == '__main__':
    # expand_dim and data_augmentation_f should be set to True together
    train_data, test_data, test_labels, AA_train_labels, AA_test_labels, Sm_train_labels, Sm_test_labels, Pg_train_labels, Pg_test_labels = datasets.data_preprocess(n_dim=35, expand_dim=True)
    train(train_data, test_data, test_labels, AA_train_labels, AA_test_labels, Sm_train_labels, Sm_test_labels, Pg_train_labels, Pg_test_labels, data_aug=data_aug_list)
