import os
import math
import glob
import random
import pandas as pd 
import matplotlib.pyplot as plt

def loadData(path):
    # path of the folder containg csv files
    # currently just concatenating the list of dictionaries together
    data = []
    for fname in glob.glob(os.path.join(path, '*.csv')): 
        with open(fname, 'r') as f: 
            temp = pd.read_csv(fname).to_dict(orient='records')
            data += temp
            f.close()
    # do data processing here
    return data    

def entropy(P):
    # P is the probability distribution, it can be any iterable object
    tot = 0
    for p in P:
        tot -= (p*math.log2(p))
    return tot

def split():
    pass

def plot():
    pass

def printTree():
    pass