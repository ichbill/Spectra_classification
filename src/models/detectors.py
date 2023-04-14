#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import os
import re
import pickle
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F

def writelog(instring, filepath):
    with open(filepath,'a') as f:
        f.write(instring+'\n')
    print(instring)

class spectra_model(nn.Module):
    def __init__(self, n_dim, hidden_size, dropout_rate, expandDim):
        super(spectra_model, self).__init__()
        self.expandDim = expandDim
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(n_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate))
        self.linear_sigmoid = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        if self.expandDim == True:
            x = self.flatten(x)
        x = self.linear_relu(x)
        x = self.batch_norm(x)
        x = self.linear_sigmoid(x)

        return x

class image_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 117 * 157, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x