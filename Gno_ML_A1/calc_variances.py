import os 
import copy
import glob
import math
import scipy
import pandas as pd
from scipy.stats import variation

PATH = "./data/SignalData"
NAN = float('nan')
columns = {'filename':''}
columns.update({'electrode '+str(k):0 for k in range(14)})
variance_file = [copy.deepcopy(columns) for i in range(len(os.listdir(PATH)))]
timeseries = [[] for i in range(16)]

for k, fname in enumerate(glob.glob(os.path.join(PATH, '*.csv'))): 
    variance_file[k]['filename'] = fname.split('/')[-1]
    print(k+1, fname.split('/')[-1])

    with open(fname, 'r') as f: 
        for i, line in enumerate(f):
            if i>0:
                for j, datapoint in enumerate(line.split(',')):
                    try:
                        datapoint = float(datapoint)
                    except ValueError:
                        datapoint = NAN
                    timeseries[j].append(datapoint)
        f.close()

    variances = [0 for i in range(14)]

    for i in range(2, 16):
        variances[i-2] = variation(a=timeseries[i], nan_policy='omit')
        variance_file[k]['electrode '+str(i-2)] = variances[i-2] 
    
pd.DataFrame(variance_file).to_csv('./data/signal_variances.csv')

    

