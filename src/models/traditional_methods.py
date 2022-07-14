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

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

model_path = "/home/exx/Spoon/spectra_classification/model/"

normalized_datapath = '/home/exx/Spoon/spectra_classification/data/normalized_data.txt'

log_path = '/home/exx/Spoon/spectra_classification/logs/data_augmentation.txt'

# species_dict = {0:'AA', 1:'Fn', 2:'Pg'}
clf_names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

def writelog(instring, filepath):
    with open(filepath,'a') as f:
        f.write(instring+'\n')
    print(instring)

def default_search(train_data, train_labels, test_data, test_labels, n_dim, max_score):
    for name, clf in zip(clf_names, classifiers):
        c_time = time.time()
        classifier = MultiOutputClassifier(clf).fit(train_data, train_labels)
        scores = classifier.score(test_data, test_labels)
        if scores>=max_score:
            max_score = scores
            writelog(str(scores)+'\t'+name+'\t'+'n_dim:'+str(n_dim), linear_yaxis_path)
        print(time.time()-c_time)
    return max_score

def LinearSVM_search(train_data, train_labels, test_data, test_labels, n_dim, max_score):
    for C in np.arange(0.005, 0.1, 0.005):
        c_time = time.time()
        clf = MultiOutputClassifier(SVC(C=C,kernel='linear')).fit(train_data, train_labels)
        scores = clf.score(test_data, test_labels)
        if scores>=max_score:
            max_score = scores
            writelog(str(scores)+'\t'+'n_dim:'+str(n_dim)+
                '\t'+'C:'+str(C)+'\t'+str(time.time()-c_time), linear_yaxis_path)
        print(time.time()-c_time)
    return max_score

def NN_search(train_data, train_labels, test_data, test_labels, n_dim, max_score):
    for activation in ['identity','logistic','tanh','relu']:
        for solver in ['lbfgs','sgd','adam']:
            for alpha in [0.00001,0.0001,0.001,0.01,0.1]:
                c_time = time.time()
                clf = MLPClassifier(activation=activation, solver=solver, alpha=alpha, early_stopping=True, validation_fraction=0.25, max_iter=10000).fit(train_data, train_labels)
                scores = clf.score(test_data, test_labels)
                if scores>=max_score:
                    max_score = scores
                    writelog(str(scores)+'\t'+'n_dim:'+str(n_dim)+'\t'
                        +'activation:'+str(activation)+'\t'
                        +'solver:'+str(solver)+'\t'
                        +'alpha:'+str(alpha)+'\t'+'early_stopping', linear_yaxis_path)
                print(time.time()-c_time)
    return max_score

def tracebyname(data_list, names_list, name):
    for i in range(len(names_list)):
        # print(names_list[i],name)
        if names_list[i] == name:
            return data_list[i]