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

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from tensorflow import keras

AA_data_path = "/home/exx/Spoon/spectra_classification/data/03-10-2022/AA/"
Sm_data_path = "/home/exx/Spoon/spectra_classification/data/03-10-2022/Sm/"
Pg_data_path = "/home/exx/Spoon/spectra_classification/data/03-10-2022/Pg/"

AA_Pg_data_path = "/home/exx/Spoon/spectra_classification/data/Mixes_04-07-2022/AA-Pg/"
AA_Sm_data_path = "/home/exx/Spoon/spectra_classification/data/Mixes_04-07-2022/AA-Sm/"
Pg_Sm_data_path = "/home/exx/Spoon/spectra_classification/data/Mixes_04-07-2022/Pg-Sm/"

mixed_data_path = "/home/exx/Spoon/spectra_classification/data/Mixes_04-07-2022/All_3/"

normalized_datapath = '/home/exx/Spoon/spectra_classification/data/normalized_data.txt'

def writelog(instring, filepath):
    with open(filepath,'a') as f:
        f.write(instring+'\n')
    print(instring)

def hash_labels(label):
    if np.array_equal(np.array([0,0,0]),label):
        return "empty species"
    elif np.array_equal(np.array([1,0,0]),label):
        return "AA"
    elif np.array_equal(np.array([0,1,0]),label):
        return "Sm"
    elif np.array_equal(np.array([0,0,1]),label):
        return "Pg"
    elif np.array_equal(np.array([1,0,1]),label):
        return "AA_Pg"
    elif np.array_equal(np.array([1,1,0]),label):
        return "AA_Sm"
    elif np.array_equal(np.array([0,1,1]),label):
        return "Sm_Pg"
    elif np.array_equal(np.array([1,1,1]),label):
        return "AA_Sm_Pg"

def one_hot_encoding(label):
    if label=="empty species":
        return np.array([0,0,0])
    elif label=="AA":
        return np.array([1,0,0])
    elif label=="Sm":
        return np.array([0,1,0])
    elif label=="Pg":
        return np.array([0,0,1])
    elif label=="AA_Pg":
        return np.array([1,0,1])
    elif label=="AA_Sm":
        return np.array([1,1,0])
    elif label=="Sm_Pg":
        return np.array([0,1,1])
    elif label=="AA_Sm_Pg":
        return np.array([1,1,1])

def get_label(filename):
    label = np.array([0,0,0])
    if ("aa" in filename) or ("AA" in filename):
        label[0] = 1
    if ("-u-30s" in filename) or ("u-30s" in filename) or ("ua-30sec" in filename):
        label[1] = 1
    if ("pg" in filename) or ("Pg" in filename):
        label[2] = 1
    elif "all3mix" in filename:
        label[0:3] = 1
    return label

def read_data(data_path):
    samples = []
    labels = []
    filenames = []
    files = os.listdir(data_path)
    for file in sorted(files):
        label = get_label(file)
        # print(file, label)
        with open(data_path+file) as f:
            lines = f.readlines()
        if '#' in lines[0]:
            lines.pop(0)
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n','')
            lines[i] = lines[i].split('\t')
            for j in range(len(lines[i])):
                lines[i][j] = float(lines[i][j])
        sample = np.array(lines)
        filenames.append(file)
        samples.append(sample)
        labels.append(label)
    return samples, labels, filenames

def shuffle_data(data, target, filename):
    idx = np.random.permutation(len(data))
    shuffled_data = data[idx]
    shuffled_target = target[idx]
    shuffled_filename = filename[idx]
    return shuffled_data, shuffled_target, shuffled_filename

# extract specific species data
def separate_data_by_species(mixed_data, target_labels, label, target_names):
    separated_data = []
    separated_label = []
    separated_names = []
    for idx in range(len(target_labels)):
        if target_labels[idx] == label:
            species_data = mixed_data[idx]
            separated_data.append(species_data)
            separated_label.append(target_labels[idx])
            separated_names.append(target_names[idx])
    separated_data = np.array(separated_data)
    separated_label = np.array(separated_label)
    separated_names = np.array(separated_names)
    separated_data, separated_label, separated_names = shuffle_data(separated_data, separated_label, separated_names)
    return separated_data, separated_label, separated_names

def train_val_test_split(Mixed_data, target_labels, target_names, train_size):
    target_labels = [hash_labels(x) for x in target_labels]
    species_types = list(set(target_labels))
    print(f'{species_types=}')
    
    train_data = []
    train_labels = []
    train_files = []
    test_data = []
    test_labels = []
    test_files = []
    for species in species_types:
        species_data, species_labels, species_files = separate_data_by_species(Mixed_data, target_labels, species, target_names)
        species_data = list(species_data)
        species_labels = list(species_labels)
        species_files = list(species_files)
        
        species_train = species_data[:int(train_size*len(species_data))]
        species_test = species_data[int(train_size*len(species_data)):]
        
        train_data = train_data + species_train
        train_labels = train_labels + species_labels[:int(train_size*len(species_data))]
        train_files = train_files + species_files[:int(train_size*len(species_data))]
        test_data = test_data + species_test
        test_labels = test_labels + species_labels[int(train_size*len(species_data)):]
        test_files = test_files + species_files[int(train_size*len(species_data)):]
        
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    
    train_labels = [one_hot_encoding(x) for x in train_labels]
    test_labels = [one_hot_encoding(x) for x in test_labels]
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    train_files = np.array(train_files)
    test_files = np.array(test_files)
    
    train_data, train_labels, train_files = shuffle_data(train_data, train_labels, train_files)
    test_data, test_labels, test_files = shuffle_data(test_data, test_labels, test_files)
    
    # print("number of training data:",len(train_data))
    # print("number of test data:",len(test_data))
    
    return train_data, train_labels, train_files, test_data, test_labels, test_files

def generate_binary_labels(train_labels, test_labels):
    train_binary_labels = []
    test_binary_labels = []
    for label in train_labels:
        if sum(label) > 1:
            train_binary_labels.append('mixed')
        else:
            train_binary_labels.append('pure')
    
    for label in test_labels:
        if sum(label) > 1:
            test_binary_labels.append('mixed')
        else:
            test_binary_labels.append('pure') 
            
    return np.array(train_binary_labels), np.array(test_binary_labels)

def get_test_data_labels(y_pred,test_labels,test_data):
    pred_pure_test_data = []
    pred_pure_test_labels = []
    pred_mixed_test_data = []
    pred_mixed_test_labels = []
    
    for i in range(len(y_pred)):
        if y_pred[i] == 'pure':
            pred_pure_test_data.append(test_data[i])
            pred_pure_test_labels.append(test_labels[i])
        else:
            pred_mixed_test_data.append(test_data[i])
            pred_mixed_test_labels.append(test_labels[i])
            
    pred_pure_test_labels = [hash_labels(x) for x in pred_pure_test_labels]
    
    return np.array(pred_pure_test_data), np.array(pred_pure_test_labels), np.array(pred_mixed_test_data), np.array(pred_mixed_test_labels), 

def label_trans(labels):
    label_list = []
    for label in labels:
        trans_label = hash_labels(label)
        label_list.append(trans_label)
    return label_list

def pca_data(all_data, n_dim):
    all_data_flatten = []
    for species_data in all_data:
        species_data_y = species_data[:,1]
        all_data_flatten.append(species_data_y)
    all_data_flatten = np.array(all_data_flatten)
    all_data_flatten = preprocessing.StandardScaler().fit_transform(all_data_flatten)

    # save normalized data
    # save_normalized_data(all_data_flatten, target_labels, normalized_datapath)

    if n_dim<1:
        return all_data_flatten
    else:
        pca = PCA(n_components=n_dim)
        all_data_pca = pca.fit(all_data_flatten).transform(all_data_flatten)
        print(f'{all_data_pca.shape=}')
        return all_data_pca

def plot_spectra(data, names, plot_path):
    x = data[:,0]
    y = data[:,1]

    plt.plot(x, y, 'b-', linewidth=1)
    plt.title(str(names)+'_l_filter')
    plt.grid()
    # print(names.split('.')[0])
    plt.savefig(plot_path+names.split('.')[0]+'_l_filter', format='png')
    plt.cla()

def tracebyname(data_list, names_list, name):
    for i in range(len(names_list)):
        # print(names_list[i],name)
        if names_list[i] == name:
            return data_list[i]

def save_normalized_data(data, labels, data_path):
    with open(data_path,'w') as f:
        for i, spectra in enumerate(data):
            for intensity in spectra:
                f.write(str(intensity)+' ')
            f.write(hash_labels(labels[i])+'\n')

def data_preprocess(n_dim=35, expand_dim=True):
    # read data
    AA_data, AA_labels, AA_file_names = read_data(AA_data_path)
    print('Number of AA samples:',len(AA_data))
    Pg_data, Pg_labels, Pg_file_names = read_data(Pg_data_path)
    print('Number of Pg samples:',len(Pg_data))
    Sm_data, Sm_labels, Sm_file_names = read_data(Sm_data_path)
    print('Number of Sm samples:',len(Sm_data))
    mixed_data, mixed_labels, mixed_file_names = read_data(mixed_data_path)
    print('Number of mixed samples:',len(mixed_data))
    AA_Pg_data, AA_Pg_labels, AA_Pg_file_names = read_data(AA_Pg_data_path)
    print('Number of AA-Pg samples:',len(AA_Pg_data))
    AA_Sm_data, AA_Sm_labels, AA_Sm_file_names = read_data(AA_Sm_data_path)
    print('Number of AA-Sm samples:',len(AA_Sm_data))
    Pg_Sm_data, Pg_Sm_labels, Pg_Sm_file_names = read_data(Pg_Sm_data_path)
    print('Number of Pg-Sm samples:',len(Pg_Sm_data))

    all_data = AA_data + Sm_data + Pg_data + AA_Pg_data + AA_Sm_data + Pg_Sm_data + mixed_data
    target_labels = AA_labels + Sm_labels + Pg_labels + AA_Pg_labels + AA_Sm_labels + Pg_Sm_labels + mixed_labels
    target_names = AA_file_names + Sm_file_names + Pg_file_names + AA_Pg_file_names + AA_Sm_file_names + Pg_Sm_file_names + mixed_file_names
    print('Number of samples:', len(all_data))
    print('Data dimensions:', len(all_data[0]))

    all_data_pca = pca_data(all_data, n_dim)
    # save normalized data
    # save_normalized_data(all_data_flatten, target_labels, normalized_datapath)

    train_data, train_labels, train_files, test_data, test_labels, test_files = train_val_test_split(all_data_pca, target_labels, target_names, 0.75)
    AA_train_labels = np.array(train_labels)[:,0]
    Sm_train_labels = np.array(train_labels)[:,1]
    Pg_train_labels = np.array(train_labels)[:,2]

    AA_test_labels = np.array(test_labels)[:,0]
    Sm_test_labels = np.array(test_labels)[:,1]
    Pg_test_labels = np.array(test_labels)[:,2]

    if expand_dim == True:
        train_data = np.array([np.expand_dims(x,axis=1) for x in train_data])
    print(f'{train_data.shape=}')

    return train_data, test_data, test_labels, AA_train_labels, AA_test_labels, Sm_train_labels, Sm_test_labels, Pg_train_labels, Pg_test_labels
