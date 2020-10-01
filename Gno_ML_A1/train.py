import pprint
import random
import argparse
from src.model import Tree
from src.utils import loadData, accuracy, splitData

ITERS = 10
SPLIT_RATIO = 0.8
MAX_DEPTH = 12
DEPTH_UPPER = 7
DEPTH_LOWER = 5
PATH = './data/final_dataset.csv'

data, target, attr = loadData(path=PATH)
random.seed(1)
seeds = random.sample([i for i in range(ITERS*10)], ITERS)
print(seeds)
model = Tree(attr)
model.growTree(data, max_d=MAX_DEPTH)
# pprint.pprint(model.tree, width=1)
# model.printTree()
scores = []
for depth in range(DEPTH_LOWER, DEPTH_UPPER+1):
    score = 0
    for seed in seeds:
        train_data, test_data = splitData(data=data, SEED=seed, ratio=SPLIT_RATIO)
        temp_model = Tree(attr)
        temp_model.growTree(train_data, max_d=depth)
        temp = accuracy(test_data, temp_model)
        print(temp)
        score += temp
    score /= ITERS
    print("depth={}, accuracy={}".format(depth, score))
    scores.append(score)
print(scores)
