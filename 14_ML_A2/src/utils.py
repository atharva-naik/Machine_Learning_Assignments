#!/usr/bin/env python3
import os
import copy
import math
import random
import warnings
import numpy as np
import pandas as pd
# from .model import cprint
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def cprint(text, r=255, g=0, b=0):
    print("\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text))
# from sklearn import cross_validation

# Load data loads the df, replaces NAN values, encodes categorical values returns a matrix of features and feature names, and also stores it as a csv
# fill_method is mean or mode
def loadData(path='data/Train_D.csv', fillin=None, fill_method=None, encode=None, drop=None, clean=False, save=False, target_field='') :
    dataset = pd.read_csv(path)
    print(f'loaded dataset, {len(dataset)} entires read ...')
    columns = dataset.columns
    print(f"columns are :\n{list(columns)}")
    print("\nfilling NAN values for :")
    for i, item in enumerate(fillin) :
        if fill_method[i] == 'mean' :
            print(f"{item}, mean={dataset[item].mean()}")
            dataset[item].fillna(dataset[item].mean(), inplace=True)
        elif fill_method[i] == 'mode' :
            print(f"{item}, has mode: {dataset[item].mode().values[0]}")
            dataset[item].fillna(dataset[item].mode().values[0], inplace=True)
        else :
            raise(Exception("Not a valid strategy !, choose out of ['mean','mode']"))
    print()

    # rectify error in 'death' label, i.e. the target. specific to this dataset
    if clean :
        print("cleaning up dataset ...")
        for i,_ in enumerate(dataset['death']) :
            if dataset['death'][i] != '0' :
                dataset['death'][i] = '1'
    print("dropping following columns :")
    for i, item in enumerate(drop) :
        try :
            print(f"{i}. {item}")
            dataset.drop(item, 1, inplace=True)
        except :
            print()
            warnings.warn(f"column {item} doesn't exit !")
            print()
    print()

    # encode categorical features
    for feature in encode :
        values = set(dataset[feature])
        for value in values :
            temp = []
            for item in dataset[feature] :
                if item == value :
                    temp.append(1)
                else :
                    temp.append(0)
            dataset[feature+'='+value] = temp
        dataset = dataset.drop(feature, 1)
    print()

    # save dataset if flag is true
    if save :
        try : 
            os.mkdir('data')
        except FileExistsError :
            print()
            warnings.warn(f"{os.getcwd()+'/data'} already exists !")
            print()
        dataset.to_csv('data/features.csv', index=False)
        print(f"features saved as {os.getcwd()+'/data/features.csv'} ...")
    print()

    target = np.array(list(dataset[target_field]), dtype=float)
    dataset.drop(target_field, 1, inplace=True)
    feature_names = list(dataset.columns)
    feature_matrix = np.array(dataset[:].values, dtype=float) 

    return feature_matrix, feature_names, target

def removeOutliers(feature_matrix, k=3, thresh=2, max_removed=100, verbose=False) :
    init_size = len(feature_matrix)
    curr_max = len(feature_matrix)
    removed = 0
    print("REMOVING OUTLIERS :")
    while curr_max >= thresh and removed < max_removed :  
        tp = np.transpose(feature_matrix)
        means, variances = [], []
        for feature in tp :
            means.append(np.mean(feature))
            variances.append(np.var(feature))
        num_oultier_features = []
        for i, point in enumerate(feature_matrix) :
            num_oultier_features.append(0)
            for j, feature in enumerate(point) :
                if abs(feature-means[j])/(math.sqrt(variances[j])+1e-12) >= k :
                    num_oultier_features[-1] += 1
        feature_matrix = np.delete(feature_matrix, num_oultier_features.index(max(num_oultier_features)), 0)
        curr_max = max(num_oultier_features)
        removed += 1
        if verbose :
            print(f"max number of outlier features={max(num_oultier_features)} @ index={num_oultier_features.index(max(num_oultier_features))}")
    cprint(f"{init_size-len(feature_matrix)} entries dropped !")
    return np.array(feature_matrix, dtype=float)

def getAccuracy(predictions, true_values, norm=True, verbose=False) :
    count = 0
    for x, y in zip(predictions, true_values) :
        if int(x) == int(y) :
            count += 1
    # print false positive and true negatives if it is binary
    if verbose :
        fp, fn = 0, 0
        if len(set(true_values)) <= 2 and len(set(predictions)) <= 2 :
            for x, y in zip(predictions, true_values) :
                if x == 1 and y == 0 :
                    fp += 1
                elif x == 0 and y == 1 :
                    fn += 1
            print(f"false positives={fp} true negatives={fn}")
    # also print class wise accuracy
    classes = set(predictions).union(set(true_values))
    if verbose :
        print(f"Following classes found :\n{classes}")
    
    for class_ in classes :
        accuracy_for_a_class = 0
        ctr = 0
        for x, y in zip(predictions, true_values) :
            if y == class_:
                ctr += 1
                if x == y :
                    accuracy_for_a_class += 1
        if ctr == 0 :
            if verbose :
                print(f"class {class_} not in true values")
        else :
            if verbose : 
                print(f"accuracy of class {class_}={accuracy_for_a_class/ctr}")

    if norm :
        return count/len(true_values)
    else :
        return count


# Plots the Cummulative variance vs no. of components graph
def plotPCAVariance(pca, X, name):
    percentage_var_explained = pca.explained_variance_ratio_;  
    cum_var_explained = np.cumsum(percentage_var_explained) #plot PCA spectrum   
    plt.figure(1, figsize = (6,4))
    plt.clf()  
    plt.plot(cum_var_explained,linewidth = 2)  
    plt.axis('tight')  
    plt.grid() 
    plt.xlabel('Number of components') 
    plt.ylabel('Cumulative Variance explained') 
    plt.title('No. of components vs. Variance')
    plt.savefig(name, dpi = 300, bbox_inches = 'tight')
    # Saves the plot
    print(f'Plot saved as {name}')
    
# Principal component analysis on feature matrix keeping some amount of variance preserved
# Takes as argument the train set and the the variance of the train set it preserves

# Usage
# reduced_matrix, dim, scaler, pca = PCA_variance(X_train, 0.95)
# test_standaradized = scaler.transform(X_test) ---> standardizes the test set
# pca_test = pca.transform(test_standardized) ---> applies PCA learnt on test set

def PCA_variance(train, test, name, variance = 0.95):
    # Standaradizing the data
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    
    # 95% of variance
    pca = PCA(n_components = variance, svd_solver = 'full')
    train = pca.fit_transform(train)
    test = scaler.transform(test)
    test = pca.transform(test)

    # the number of reduced features
    dim = np.shape(train)[1]
    print('Number of components to explain', variance*100, '% variance is', dim)
    # plots the graph
    plotPCAVariance(pca, train, name=name)
    # returns the reduced matrix of features, number of reduced features, 
    # the standard scaler fitted to the train set, and the pca fitted to the train set 
    return train, test, dim


# def splitData(data, target, seed=666, ratio=0.2):
#     copy_data = copy.deepcopy(data)
#     copy_target = copy.deepcopy(target)
#     test_size = int(len(copy_data)*ratio)
    
#     random.Random(seed).shuffle(copy_data)
#     test = copy_data[ : test_size]
#     train = copy_data[test_size : ]   
#     test_t = copy_target[ : test_size]
#     train_t = copy_target[test_size : ]  

#     return train, train_t, test, test_t

