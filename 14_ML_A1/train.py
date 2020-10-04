import copy
import pprint
import random
import argparse
from src.model import Tree
from src.utils import loadData, accuracy, splitData, plotDepthVAccuracy, cprint

## terminal arguments ##
parser = argparse.ArgumentParser()
parser.add_argument("-i","--iters", type=int, default=10, help="number of iterations to determine accuracy")
parser.add_argument("-t","--test_size", type=float, default=0.2, help="ratio of test data to total")
parser.add_argument("-v","--val_size", type=float, default=0.2, help="ratio of validation data to total")
parser.add_argument("-p","--path", type=str, default='./data/final_dataset.csv', help="path to dataset")
parser.add_argument("-u","--upper", type=int, default=16, help="maximum depth value to be explored")
parser.add_argument("-d","--max_depth", type=int, default=-1, help="maximum depth value of solo decision tree")
parser.add_argument("-l","--lower", type=int, default=2, help="minimum depth value to be explored")
args = parser.parse_args()

ITERS = args.iters
TEST_SIZE = args.test_size
VAL_SIZE = args.val_size
MAX_DEPTH = args.max_depth
DEPTH_UPPER = args.upper
DEPTH_LOWER = args.lower
PATH = args.path

data, target, attr = loadData(path=PATH)
random.seed(1)
seeds = sorted(random.sample([i for i in range(ITERS*50)], ITERS))
print("Seeds used for 10 splits are = {}".format(seeds))
# model = Tree(attr)
# model.growTree(data, max_d=MAX_DEPTH)
# # pprint.pprint(model.tree, width=1)
# # model.printTreeTerminal()
# print(accuracy(model=model, data=data))
# # model.printTreeImage()
cprint("\nDETERMING BEST DEPTH VALUE (PRE PRUNING/ EARLY STOP) ...\n",r=255,b=255,g=0)
scores = []
best_accuracy, best_depth = 0, DEPTH_LOWER
best_tree = None
best_split = {'test':[],'train':[],'val':[]}
depths = [i for i in range(DEPTH_LOWER, DEPTH_UPPER+1)]
for depth in depths:
    score = 0
    best_accuracy_for_this_depth = 0
    best_tree_for_this_depth = None
    best_split_for_this_depth = {'test':[],'train':[],'val':[]}
    for seed in seeds:
        train, test, val = splitData(data=data, seed=seed, val_size=VAL_SIZE, test_size=TEST_SIZE)
        temp_model = Tree(attr)
        temp_model.growTree(train, max_d=depth)
        temp = accuracy(test, temp_model, norm=1)
        
        if best_accuracy_for_this_depth < temp:
            best_accuracy_for_this_depth = temp
            best_tree_for_this_depth = temp_model
            best_split_for_this_depth['train'] = train
            best_split_for_this_depth['test'] = test
            best_split_for_this_depth['val'] = val
        score += temp
    
    score /= ITERS
    if score >= best_accuracy:
        best_accuracy = score
        best_split = best_split_for_this_depth
        best_depth = depth
        best_tree = best_tree_for_this_depth
    cprint("depth={} accuracy={}".format(depth, score), r=255, g=255)
    scores.append(score)


cprint("\nBest depth is {}".format(best_depth),b=255)
cprint("\nPLOTTING DEPTH VS ACCURACY ...\n",b=100)
plotDepthVAccuracy(DEPTH_LOWER, DEPTH_UPPER, scores, ITERS)
cprint("Accuracy of best tree on test set before pruning = {}".format(accuracy(best_split['test'], best_tree)), r=255,b=255,g=0)
# best_tree.prune()
best_tree.printTreeTerminal()
cprint("\nSAVING IMAGE OF TREE BEFORE PRUNING ...\n",r=180,b=180,g=180)
print("validation set accuracy before pruning = {}".format(accuracy(best_split['val'], best_tree)))
best_tree.printTreeImage()
cprint("\nPRUNING THE BEST TREE ...",r=255,g=160,b=0)
best_tree.postPrune(best_split['val'])
cprint("\nSAVING IMAGE OF TREE AFTER PRUNING ...\n",r=180,b=180,g=180)
best_tree.printTreeImage('pruned_decision_tree')
print("validation set accuracy after pruning = {}".format(accuracy(best_split['val'], best_tree)))
cprint("Accuracy of best tree on test set after pruning = {}".format(accuracy(best_split['test'], best_tree)), r=255,b=255,g=0)


## Showing a case to validate pruning procedure ##
cprint("\nA CASE WHERE PRUNING IMPROVES PERFORMANCE\n",r=255,g=0)
cprint("\ndepth=5",b=255)
model = Tree(attr)
model.growTree(best_split['train'], max_d=5)
print("test set accuracy before pruning = {}".format(accuracy(best_split['test'], model)))
print("validation set accuracy before pruning = {}".format(accuracy(best_split['val'], model)))
model.postPrune(best_split['val'])
print("validation set accuracy after pruning = {}".format(accuracy(best_split['val'], model)))
print("test set accuracy after pruning = {}".format(accuracy(best_split['test'], model)))