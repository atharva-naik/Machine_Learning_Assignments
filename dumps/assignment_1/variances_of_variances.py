import math
import scipy
import pandas as pd
from scipy.stats import variation

df = pd.read_csv('./data/signal_variances.csv', keep_default_na=False) 
variances = [0 for i in range(14)]
k = 0 
for i in range(14):
    temp = []
    for j, val in enumerate(df['electrode '+str(i)]): 
        if df['test-type'][j] != '':
            k += 1
            temp.append(val)
    variances[i] = variation(a=temp, nan_policy='omit')
print(variances)
print(k/14)

