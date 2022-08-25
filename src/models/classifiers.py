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

from data_augmentation import augusage as aug
from utils import datasets
from utils import plot
from models import traditional_methods

model_path = "/home/exx/Spoon/spectra_classification/model/"

normalized_datapath = '/home/exx/Spoon/spectra_classification/data/normalized_data.txt'

log_path = '/home/exx/Spoon/spectra_classification/logs/data_augmentation.txt'

# species_dict = {0:'AA', 1:'Fn', 2:'Pg'}

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

def data_augmentation(data, labels, aug_list):
    if len(aug_list)==0:
        return data, labels
    # else:
    #     print('using data augmentation methods ...')
    #     print(f'{aug_list}')

    aug_data = []
    aug_labels = []
    for i, spectra in enumerate(data):
        jitter_spec = aug.jitter(spectra)
        permutation_spec = aug.permutation(spectra)
        aug_data.append(spectra)
        aug_labels.append(labels[i])
        if 'jitter' in aug_list:
            aug_data.append(jitter_spec)
            aug_labels.append(labels[i])
        if 'permutation' in aug_list:
            aug_data.append(permutation_spec)
            aug_labels.append(labels[i])

    aug_data = np.array(aug_data)
    aug_labels = np.array(aug_labels)

    if 'spawner' in aug_list:
        spawner_spec = aug.spawner(data, labels)
        aug_data = np.concatenate((aug_data,spawner_spec))
        aug_labels = np.concatenate((aug_labels,labels))

    if 'wdba' in aug_list:
        wdba_spec = aug.wdba(data, labels)
        aug_data = np.concatenate((aug_data,wdba_spec))
        aug_labels = np.concatenate((aug_labels,labels))

    if 'rgw' in aug_list:
        rgw_spec = aug.random_guided_warp(data, labels)
        aug_data = np.concatenate((aug_data,rgw_spec))
        aug_labels = np.concatenate((aug_labels,labels))

    if 'dgw' in aug_list:
        dgw_spec = aug.discriminative_guided_warp(data, labels)
        aug_data = np.concatenate((aug_data,dgw_spec))
        aug_labels = np.concatenate((aug_labels,labels))

    return aug_data, aug_labels

def train_AA(train_data, AA_train_labels, test_data, AA_test_labels, ckpt_path='', data_augmentation_f=[], n_dim=35, dropout=0.45, lr=1e-2, hidden_size=100, mode='train'):
    keras.backend.clear_session()

    AA_train_data, AA_train_labels = data_augmentation(train_data, AA_train_labels, data_augmentation_f)

    AA_model = keras.models.Sequential()
    AA_model.add(keras.layers.Dense(hidden_size,activation='relu',input_shape=(n_dim,)))
    AA_model.add(keras.layers.BatchNormalization())
    AA_model.add(keras.layers.Dropout(dropout))
    AA_model.add(keras.layers.Dense(1,activation='sigmoid'))

    #print(AA_model.summary())
    learning_rate = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10000,
        decay_rate=0.7)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    AA_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if mode == 'train':
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',patience=300,verbose=0),keras.callbacks.ModelCheckpoint(filepath='../model/AA_model_ckpt.h5',monitor='val_accuracy',save_best_only=True,verbose=0)]

        history = AA_model.fit(AA_train_data, AA_train_labels, epochs=1000, batch_size=256, callbacks=callbacks, shuffle=True, validation_data=(test_data, AA_test_labels), verbose=0)

        AA_model.load_weights('../model/AA_model_ckpt.h5')
        # plot.plot_history(history)
    elif mode == 'test':
        AA_model.load_weights(ckpt_path)

    performance = AA_model.evaluate(test_data, AA_test_labels, verbose=0)
    # print(f'{performance=}')
    # writelog('AA accuracy:'+str(performance[1]), log_path)

    AA_results = AA_model.predict(test_data, verbose=0)
    for i in range(len(AA_results)):
        AA_results[i] = round(AA_results[i][0])
    # print(f"{AA_results.shape=}")

    return AA_results, performance[1], performance[0], AA_model

def train_Sm(train_data, Sm_train_labels, test_data, Sm_test_labels, ckpt_path='', data_augmentation_f=[], n_dim=35, dropout=0.45, lr=1e-2, hidden_size=100, mode='train'):
    keras.backend.clear_session()

    Sm_train_data, Sm_train_labels = data_augmentation(train_data, Sm_train_labels, data_augmentation_f)

    Sm_model = keras.models.Sequential()
    Sm_model.add(keras.layers.Dense(hidden_size,activation='relu',input_shape=(n_dim,)))
    Sm_model.add(keras.layers.BatchNormalization())
    Sm_model.add(keras.layers.Dropout(dropout))
    Sm_model.add(keras.layers.Dense(1,activation='sigmoid'))

    learning_rate = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    Sm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if mode == 'train':
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',patience=300),keras.callbacks.ModelCheckpoint(filepath='../model/Sm_model_ckpt.h5',monitor='val_accuracy',save_best_only=True)]

        history = Sm_model.fit(Sm_train_data, Sm_train_labels, epochs=1000, batch_size=256, callbacks=callbacks, shuffle=True, validation_data=(test_data, Sm_test_labels), verbose=0)
        # plot.plot_history(history)
        Sm_model.load_weights('../model/Sm_model_ckpt.h5')
    elif mode == 'test':
        Sm_model.load_weights(ckpt_path)

    performance = Sm_model.evaluate(test_data, Sm_test_labels, verbose=0)
    # print(performance)
    # writelog('Sm accuracy:'+str(performance[1]), log_path)

    Sm_results = Sm_model.predict(test_data, verbose=0)
    for i in range(len(Sm_results)):
        Sm_results[i] = round(Sm_results[i][0])
    # print(f"{Sm_results.shape=}")

    return Sm_results, performance[1], performance[0], Sm_model

def train_Pg(train_data, Pg_train_labels, test_data, Pg_test_labels, ckpt_path='', data_augmentation_f=[], n_dim=35, dropout=0.45, lr=1e-2, hidden_size=100, mode='train'):
    keras.backend.clear_session()

    Pg_train_data, Pg_train_labels = data_augmentation(train_data, Pg_train_labels, data_augmentation_f)

    Pg_model = keras.models.Sequential()
    Pg_model.add(keras.layers.Dense(hidden_size,activation='relu',input_shape=(n_dim,)))
    Pg_model.add(keras.layers.BatchNormalization())
    Pg_model.add(keras.layers.Dropout(dropout))
    Pg_model.add(keras.layers.Dense(1,activation='sigmoid'))

    learning_rate = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10000,
        decay_rate=0.7)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    Pg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if mode == 'train':
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',patience=200),keras.callbacks.ModelCheckpoint(filepath='../model/Pg_model_ckpt.h5',monitor='val_accuracy',save_best_only=True)]

        history = Pg_model.fit(Pg_train_data, Pg_train_labels, epochs=1000, batch_size=256, callbacks=callbacks, shuffle=True, validation_data=(test_data, Pg_test_labels), verbose=0)
        Pg_model.load_weights('../model/Pg_model_ckpt.h5')
    elif mode == 'test':
        Pg_model.load_weights(ckpt_path)

    performance = Pg_model.evaluate(test_data, Pg_test_labels, verbose=0)
    # print(performance)
    # writelog('Pg accuracy:'+str(performance[1]), log_path)

    Pg_results = Pg_model.predict(test_data, verbose=0)
    for i in range(len(Pg_results)):
        Pg_results[i] = round(Pg_results[i][0])
    # print(f"{Pg_results.shape=}")

    return Pg_results, performance[1], performance[0], Pg_model

