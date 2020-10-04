#!/usr/bin/env python3

import os
import copy
import math
import glob
import random
import pandas as pd
import seaborn as sns 
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

def gini_index(Q):
    tot = 1
    test = 0
    P = copy.deepcopy(Q)
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

def splitData(data, seed=0, val_size=0.2, test_size=0.2):
    copy_data = copy.deepcopy(data)
    val_size = int(len(copy_data)*val_size)
    test_size = int(len(copy_data)*test_size)
    train_size = len(copy_data) - val_size - test_size
    
    random.Random(seed).shuffle(copy_data)
    train = copy_data[ : train_size]
    test = copy_data[train_size:train_size + test_size]
    val = copy_data[train_size + test_size : ]
    
    return train, test, val

def cprint(text, r=0, g=255, b=0):
    print("\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text))

def accuracy(data, model, norm=1, verbose=0):
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

    if norm:
        return (count)/tot
    else:
        return count

def plotDepthVAccuracy(lower_depth, upper_depth, scores, iters_used=10, fname='depth_vs_accuracy_plot'):
    depth_values = [i for i in range(lower_depth, upper_depth+1)]
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (25,17)
    
    plt.plot(depth_values, scores)
    plt.xticks(depth_values, labels=depth_values)
    plt.xlabel('depth values')
    plt.ylabel('accuracy')
    plt.title('Accuracy averaged over {} splits'.format(iters_used))
    plt.savefig(fname+'.png')

    best_depth = depth_values[scores.index(max(scores))]
    cprint("plot saved as {}".format(fname+'.png'),r=255,b=100,g=100)
    cprint("best depth for accuracy averaged over {} iterations is {}".format(iters_used, best_depth),r=255,b=100,g=100)
    return best_depth