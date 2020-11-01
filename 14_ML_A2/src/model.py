#!/usr/bin/env python3
import copy
import math
import tqdm
import random
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from .utils import getAccuracy, cprint, PCA_variance

def subset(l, mask, val=0) :
    op = []
    for i, item in enumerate(l) :
        if mask[i] == val :
            op.append(item)
    return op

def logGaussPDF(mu, var, x) :
    if var == 0.0 :
        var += 1e-8
    return 0.5*math.log(2*np.pi*var) - 0.5*((x-mu)**2/var)

class nBGClassifier() : # naive bayes gaussian classifier
    def __init__(self, classes=None) :
        self.means = []
        self.variances = []
        self.class_dist = {}
        self.num_classes = 0
        self.num_features = 0
        self.classes = classes

    def fit(self, data, target) :
        self.num_features = len(data[0])
        if not(self.classes):
            self.classes = sorted(list(set(target)))
        self.num_classes = len(self.classes)
        
        data = np.array(data)
        transpose = np.transpose(data)
        
        for class_ in self.classes :
            self.class_dist[class_] = 0
        
        for item in target :
            self.class_dist[item] += 1
        norm = sum(self.class_dist.values())
        
        for class_ in self.classes :
            self.class_dist[class_] /= norm   

        for i, class_ in enumerate(self.classes) :
            self.means.append([])
            self.variances.append([])
            for j in range(self.num_features) :
                self.means[i].append(np.mean(subset(transpose[j], target, class_)))
                self.variances[i].append(np.var(subset(transpose[j], target, class_)))
    
    def classify(self, x) :
        op = []
        for item in x :  
            scores = []
            for i, class_ in enumerate(self.classes) :
                score = 0
                for j in range(self.num_features) :
                    score += logGaussPDF(self.means[i][j], self.variances[i][j], item[j])
                score += math.log(self.class_dist[class_])
                scores.append(score)
            op.append(scores.index(max(scores)))
        
        return np.array(op)
                

def evalKFold(model, data, target, splits=5, shuffle=True, random_state=666, verbose=False) :
    
    kf = KFold(n_splits=splits, shuffle=shuffle, random_state=random_state)
    accuracy = 0
    if verbose :
        print(f"    K={splits}\n    shuffle={shuffle}\n    random_sate={random_state}\n    target has length={len(target)}\n    data has length={len(data)}\n    number of features are={len(data)}\n ")
    i = 1
    for train_indices, test_indices in kf.split(data):
        if verbose :
            cprint(f"************* Fold-{i} ************* :\n")
        i += 1
        train = [data[i] for i in train_indices]
        train_target = [target[i] for i in train_indices]
        test = [data[i] for i in test_indices]
        test_target = [target[i] for i in test_indices]
        # normalise the feature matrix (passed as data)
        model.fit(train, train_target)
        preds = model.classify(test)
        
        test_dist = {}
        for class_ in model.classes :
            test_dist[class_] = 0
        for item in test_target :
            test_dist[item] += 1
        test_dist = list(test_dist.values())
    
        train_dist = {}
        for class_ in model.classes :
            train_dist[class_] = 0
        for item in train_target :
            train_dist[item] += 1
        train_dist = list(train_dist.values())

        if verbose :
            print(f"train_size={len(train_indices)} test_size={len(test_indices)}")
            cprint(f"IN TRAIN : minority class is {train_dist.index(min(train_dist))} has {train_dist[train_dist.index(min(train_dist))]} points",r=54, g=115, b=119)
            cprint(f"IN TEST : minority class is {test_dist.index(min(test_dist))} has {test_dist[test_dist.index(min(test_dist))]} points",r=54, g=115, b=119)
            cprint(f"accuracy={getAccuracy(preds, test_target)}",r=170, g=255)
        accuracy += getAccuracy(preds, test_target, norm=False, verbose=verbose)
        if verbose :
            print("\n")
    accuracy = accuracy/(splits*len(test_target))
    if verbose :
        cprint(f"Net accuracy is {accuracy} ...", g=255)
    
    return accuracy

def evalKFoldPCA(model, data, target, splits=5, shuffle=True, random_state=666, verbose=False) :
    
    kf = KFold(n_splits=splits, shuffle=shuffle, random_state=random_state)
    accuracy = 0
    if verbose :
        print(f"    K={splits}\n    shuffle={shuffle}\n    random_sate={random_state}\n    target has length={len(target)}\n    data has length={len(data)}\n    number of features are={len(data)}\n ")
    j = 1
    for train_indices, test_indices in kf.split(data):
        if verbose :
            cprint(f"************* Fold-{j} ************* :\n")
        train = [data[i] for i in train_indices]
        train_target = [target[i] for i in train_indices]
        test = [data[i] for i in test_indices]
        test_target = [target[i] for i in test_indices]
        # normalise the feature matrix (passed as data)
        train, test, dim = PCA_variance(train, test, name=f'plot({j}).png')
        j += 1
        model.fit(train, train_target)
        preds = model.classify(test)
        
        test_dist = {}
        for class_ in model.classes :
            test_dist[class_] = 0
        for item in test_target :
            test_dist[item] += 1
        test_dist = list(test_dist.values())
    
        train_dist = {}
        for class_ in model.classes :
            train_dist[class_] = 0
        for item in train_target :
            train_dist[item] += 1
        train_dist = list(train_dist.values())

        if verbose :
            print(f"train_size={len(train_indices)} test_size={len(test_indices)}")
            cprint(f"IN TRAIN : minority class is {train_dist.index(min(train_dist))} has {train_dist[train_dist.index(min(train_dist))]} points",r=54, g=115, b=119)
            cprint(f"IN TEST : minority class is {test_dist.index(min(test_dist))} has {test_dist[test_dist.index(min(test_dist))]} points",r=54, g=115, b=119)
            cprint(f"accuracy={getAccuracy(preds, test_target)}",r=170, g=255)
        accuracy += getAccuracy(preds, test_target, norm=False, verbose=verbose)
        if verbose :
            print("\n")
    accuracy = accuracy/(splits*len(test_target))
    if verbose :
        cprint(f"Net accuracy is {accuracy} ...", g=255)
    
    return accuracy

def seqBackSelect(model, data, target, splits=5, shuffle=True, random_state=666) :
    # remove useless features
    before_acc = evalKFold(model, data, target, splits, shuffle, random_state)
    orig_features = data.shape[1]

    scores = []
    for i in tqdm.tqdm(data[0]) :
        temp = data
        temp = np.delete(temp, i, 1)
        scores.append(evalKFold(model, temp, target, splits, shuffle, random_state))

    after_acc = max(scores)
    if after_acc >= before_acc :
        print(f"features dropped from {orig_features} to {temp.shape(1)} \naccuracy improved from {before_acc} to {after_acc}")
        ind = scores.index(max(after_acc))
        data = np.delete(data, ind, 1)
    else :
        print(f"{after_acc} was less than {before_acc}")
        print(f"{orig_features-data.shape[1]} features removed !")
        return data