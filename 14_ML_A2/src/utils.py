import os
import copy
import math
import random
import warnings
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

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

def removeOutliers(feature_matrix, k=3, thresh=2, verbose=False) :
    curr_max = len(feature_matrix)
    while curr_max >= thresh :  
        tp = np.transpose(feature_matrix)
        means, variances = [], []
        for feature in tp :
            means.append(np.mean(feature))
            variances.append(np.var(feature))
        num_oultier_features = []
        for i, point in enumerate(feature_matrix) :
            num_oultier_features.append(0)
            for j, feature in enumerate(point) :
                if abs(feature-means[j])/(variances[j]) >= k :
                    num_oultier_features[-1] += 1
        feature_matrix = np.delete(feature_matrix, num_oultier_features.index(max(num_oultier_features)), 0)
        curr_max = max(num_oultier_features)
        if verbose :
            print(f"max number of outlier features={max(num_oultier_features)} @ index={num_oultier_features.index(max(num_oultier_features))}")
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
            print(f"class-{class_} not in true values")
        else :
            print(f"accuracy of class-{class_}={accuracy_for_a_class/ctr}")

    if norm :
        return count/len(true_values)
    else :
        return count



# def normFeatures(feature_matrix, features, target) :
#     for i, feature in enumerate(features) :
#         if feature != target :


# def cleanUpData(data):
#     # Deal with faulty entries (impurities specific to this dataset)
#     pass 

# def processData(data, store=False) :
#     # encode the categorical variables and replace missing values
#     clean_data = pd.DataFrame()
#     if store == True :
#         clean_data.to_csv("processed_data.csv")

















# def splitData(data) :
#     pass

# def splitData(data, target, seed=None, ratio=0.2):
#     if seed == None:
#         seed = random.randint(-1000,1000)
#     combined = copy.deepcopy(list(zip(data, target)))
#     val_size = int(len(combined)*ratio)
#     train_size = len(combined) - val_size
    
#     random.Random(seed).shuffle(combined)
#     train = combined[ : train_size]
#     val = combined[train_size : ]

#     val_data = [itm[0] for itm in val]
#     val_target = [itm[1] for itm in val]
#     train_data = [itm[0] for itm in train]
#     train_target = [itm[1] for itm in train]
#     return train_data, train_target, val_data, val_target

# # NA values are ignored 
# def mean(x, nan='NA') : 
#     if len(x) == 0 :
#         raise(Exception("Mean not defined for empty object"))
#     y , nans = 0, 0
#     for item in x :
#         if item == nan :
#             nans += 1
#         else :
#             y += x
#     if nans == len(x) :
#         raise(Exception("All elements are NA, can\'t compute mean"))
#     return y/(len(x)-nans) 

# # NA values are ignored
# def variance(x, nan='NA') : 
#     var, nans = 0, 0
#     mean = mean(x, nan=nan)
#     for item in x :
#         if item == nan :
#             nans += 1
#         else : 
#             var += (item-mean)**2
#     return var/(len(x)-nans)

# def replaceNAN(x, mean):
#     for item in x :