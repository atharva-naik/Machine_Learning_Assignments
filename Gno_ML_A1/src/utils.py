#!/usr/bin/env python3

import os
import math
import glob
import random
import pandas as pd 
import matplotlib.pyplot as plt

def loadData(path, verbose=False):
    df = pd.read_csv(path, keep_default_na=False)
    df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
    labels = []
    attr = list(df.columns)
    df = df.to_dict('records')

    data = []
    flag = 1
    
    for datapoint in df:
        if datapoint['test-type'] != '':
            data.append(datapoint)
            labels.append(datapoint['test-type'])
        else:
            if flag and verbose: 
                print("Some issues with the following files:")
            flag = 0
            if verbose: 
                print(datapoint['filename'])
    if verbose: 
        print("test-type data present is for {} EEG files".format(len(data)))
    
    for i,_ in enumerate(data):
        data[i]['id'] = i
    return data, labels, attr

def gini_index(P):
    tot = 1
    test = 0
    for p in P:
        test += p
    if test == 0:
        raise Exception("not a valid probability distribution")
    
    elif test == 1:
        for p in P:
            tot -= p**2
        return tot
    
    else:
        for i, p in enumerate(P):
            P[i] = (p/test)
        for p in P:
            tot -= p**2
        return tot

def splitData(data, SEED=0, ratio=0.8):
    size = len(data)
    train_size = int(ratio*size)
    test_size = size-train_size
    random.seed(SEED)
    indices = sorted(random.sample([i for i in range(size)], test_size))
    train_data, test_data = [], []

    j = 0
    for i, datapoint in enumerate(data):
        if j <test_size:
            if indices[j] == i:
                j += 1
                test_data.append(datapoint)
            else:
                train_data.append(datapoint)
        else:
            train_data.append(datapoint)

    return train_data, test_data

def cprint(text, r=0, g=255, b=0):
    print("\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text))

def accuracy(data, model, verbose=0):
    tot = len(data)
    count = 0
    preds = model.predict(data)
    for item, pred in zip(data, preds):
        if pred[0] == item['test-type']:
            if verbose: 
                cprint(item['test-type'] + ' , ' + pred[0])
            count += 1
        else:
            if verbose:
                cprint(item['test-type'] + ' , ' + pred[0], r=255, g=0)

    return (count)/tot
        