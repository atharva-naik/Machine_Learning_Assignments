import pprint
import sklearn
import argparse
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier 
from src.utils import loadData, splitData, accuracy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, SGDClassifier

# terminal arguments 
parser = argparse.ArgumentParser()
parser.add_argument("-l","--lr", type=float, default=0.1, help="learning rate of optimizer")
# parser.add_argument("-d","--device", type=str, default="gpu", help="type of device to be used")
parser.add_argument("-H","--hidden_list", type=list, default=[], help="list of hidden layer sizes")
parser.add_argument("-p","--path", type=str, default='./data/output.csv', help="path to the dataset")
parser.add_argument("-i","--max_iters", type=int, default=1000, help="maximum epochs for model training")
parser.add_argument("-s","--seed", type=int, default=666, help="seed value for random state used by sklearn")
parser.add_argument("-w","--write", type=str, default="results.txt", help="write the output accuracy to this file")
parser.add_argument("-r","--split_ratio", type=float, default=0.2, help="the ratio of size of test set to total dataset")
parser.add_argument("-o","--optim", type=str, default='adam', help="the optimizer used to train the model, out of ['adam','sgd']")
parser.add_argument("-v","--verbose", action='store_true', default=True, help="train model in verbose mode (displays loss per epoch)")
parser.add_argument("-a","--activation", type=str, default='tanh', help="the activation used for hidden layers, out of ['relu','logistic','tanh']")
args = parser.parse_args()

assert args.optim in ['sgd','adam'], 'not a valid optimizer type'
assert args.activation in ['relu','logistic','tanh'], 'not a valid activation type'


HIDDEN_LAYER_SIZES = args.hidden_list
ACTIVATION = args.activation
LEARNING_RATE = args.lr
SPLIT_RATIO = args.split_ratio
OPTIMIZER = args.optim
SEED = args.seed

# format list correctly
temp = []
for item in HIDDEN_LAYER_SIZES :
    try :
        temp.append(int(item))
    except ValueError:
        pass
HIDDEN_LAYER_SIZES = temp


## doubts
## 1. batch size (default:200)
## 2. which activation function: logistic ?
## 3. stopping condition (when val stops improving ?)

data, target = loadData(path='data/output.csv', target_field='Class {0,1}', ret_type='np', norm=True)
# train, train_t, test, test_t = splitData(data, target, seed=SEED, ratio=SPLIT_RATIO)
train, test, train_t, test_t = train_test_split(data, target, random_state=SEED, test_size=SPLIT_RATIO)
train_dist = {}
test_dist = {}

for class_ in list(set(target)) :
    train_dist[class_] = list(train_t).count(class_)
    test_dist[class_] = list(test_t).count(class_)

print("Train dataset class distribution")
pprint.pprint(train_dist, width=1)
print("Test dataset class distribution")
pprint.pprint(test_dist, width=1)

if len(HIDDEN_LAYER_SIZES) > 0 :
    model = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, 
                        random_state=SEED,
                        verbose=args.verbose,
                        activation=ACTIVATION,
                        learning_rate='constant', 
                        learning_rate_init=LEARNING_RATE, 
                        solver=OPTIMIZER,
                        max_iter=10000,
                        shuffle=True)
else :
    model = Perceptron(random_state=SEED,
                        fit_intercept=True,
                        verbose=args.verbose,
                        eta0=LEARNING_RATE, 
                        max_iter=10000,
                        shuffle=True)
    # model = SGDClassifier(loss='perceptron',
    #                     random_state=SEED,
    #                     verbose=args.verbose,
    #                     learning_rate='constant',
    #                     eta0=LEARNING_RATE,
    #                     fit_intercept=True,
    #                     max_iter=10000,
    #                     shuffle=True)
# train
print(HIDDEN_LAYER_SIZES)
model.fit(train, train_t)
# predict
predictions = model.predict(test)
print(set(predictions))
# calculate accuracy, false positives and false negatives
acc = accuracy(predictions, test_t)
print(acc)
f = open(args.write, "a")
if len(HIDDEN_LAYER_SIZES) == 0 :
    f.write("learning rate="+str(LEARNING_RATE)+" architecture=[] optimizer=sgd seed="+str(SEED))
else :
    f.write("learning rate="+str(LEARNING_RATE)+" architecture="+str(HIDDEN_LAYER_SIZES)+" optimizer="+OPTIMIZER+" activation="+ACTIVATION+" seed="+str(SEED))
f.write(" accuracy="+str(acc)+"\n")
f.close()
