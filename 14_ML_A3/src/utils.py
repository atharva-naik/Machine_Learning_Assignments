import copy
import torch 
import random
import numpy as np
import pandas as pd

def loadData(path, target_field="", ret_type='np', norm=False) :
    df = pd.read_csv(path)
    try :
        target = df[target_field]
    except KeyError :
        raise(Exception(f"the field {target_field} isn't present in {path}"))
    df.drop([target_field], axis=1, inplace=True)
    data = df.values

    if norm == True :
        maxms = [max(val) for val in np.transpose(df.values)]
        for datapoint in data :
            datapoint /= maxms

    if ret_type == 'np' :
        pass
        # data = np.array(data)
        # target = np.array(target)
    elif ret_type == 'pt' :
        data = torch.as_tensor(data)
        target = torch.as_tensor(target)
    else :
        raise(Exception(f"return type {ret_type} is invalid, please choose one from ['np','pt'] (numpy ndarray or pytorch tensor) !"))
    
    return data, target

def splitData(data, target, seed=666, ratio=0.2) :
    copy_data = copy.deepcopy(data)
    copy_target = copy.deepcopy(target)
    test_size = int(len(copy_data)*ratio)
    
    random.Random(seed).shuffle(copy_data)
    test = copy_data[ : test_size]
    train = copy_data[test_size : ]   
    test_t = copy_target[ : test_size]
    train_t = copy_target[test_size : ]  

    print(f"size of train set={len(train)} test set={len(test)}")
    print(f"shape of train dataset is {train.shape}")
    print(f"shape of test dataset is {test.shape}")
    print(f"shape of train target is {train_t.shape}")
    print(f"shape of test target is {test_t.shape}")

    return train, train_t, test, test_t

def accuracy(predictions, labels, norm=True) :
    acc, fp, fn = 0, 0, 0
    totp = list(labels).count(1)
    totn = list(labels).count(0)
    for pred, true in zip(predictions, labels) :
        if pred == true :
            acc += 1
        elif pred == 0 :
            fn += 1
        elif pred == 1 :
            fp += 1
    print(f"there are {fp}/{totp} false positives and {fn}/{totn} false negatives")
    if norm == True :
        acc/=len(labels)
    
    return acc