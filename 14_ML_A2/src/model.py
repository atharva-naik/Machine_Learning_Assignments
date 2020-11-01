import math
import random
import sklearn
import numpy as np
import pandas as pd
from .utils import getAccuracy 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def cprint(text, r=255, g=0, b=0):
    print("\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text))

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
    
    # def classify(self, x) :
    #     try :
    #         if type(x[0]) is list or np.array :
    #             output = []
    #             for item in x :
    #                 output.append(self.classify(item))
    #                 return np.array(output)
    #     except IndexError :
    #         print(x)
    #     else :
    #         scores = []
    #         for i, class_ in enumerate(self.classes) :
    #             score = 0
    #             for j in range(self.num_features) :
    #                 score -= logGaussPDF(self.means[i][j], self.variances[i][j], x[j])
    #             score -= math.log(self.class_dist[class_])
    #             scores.append(score)
    #         return scores.index(min(scores))
                    
                

def evalKFold(model, data, target, splits=5, shuffle=True, random_state=666) :
    
    kf = KFold(n_splits=splits, shuffle=shuffle, random_state=random_state)
    accuracy = 0
    print(f"K={splits}\n shuffle={shuffle}\n random_sate={random_state}\n target has length={len(target)}\n data has length={len(data)}\n number of features are={len(data)}\n ")
    i = 1
    for train_indices, test_indices in kf.split(data):
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

        print(f"train_size={len(train_indices)} test_size={len(test_indices)}")
        print(f"IN TRAIN : minority class={train_dist.index(min(train_dist))} has {train_dist[train_dist.index(min(train_dist))]} points")
        print(f"IN TEST : minority class={test_dist.index(min(test_dist))} has {test_dist[test_dist.index(min(test_dist))]} points")
        cprint(f"accuracy={getAccuracy(preds, test_target)}",r=0,g=255)
        accuracy += getAccuracy(preds, test_target, norm=False, verbose=True)
        print("\n")
    cprint(f"Net accuracy is {accuracy/(splits*len(test_target))} ...", g=255)




