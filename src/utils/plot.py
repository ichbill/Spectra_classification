#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import os
import sys

import matplotlib.pyplot as plt

model_path = "/home/exx/Spoon/spectra_classification/model/"

normalized_datapath = '/home/exx/Spoon/spectra_classification/data/normalized_data.txt'

log_path = '/home/exx/Spoon/spectra_classification/logs/data_augmentation.txt'

def plot_spectra(data, names, plot_path):
    x = data[:,0]
    y = data[:,1]

    plt.plot(x, y, 'b-', linewidth=1)
    plt.title(str(names)+'_l_filter')
    plt.grid()
    # print(names.split('.')[0])
    plt.savefig(plot_path+names.split('.')[0]+'_l_filter', format='png')
    plt.cla()

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1,len(acc)+1)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()