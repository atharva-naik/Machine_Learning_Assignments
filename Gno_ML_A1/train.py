import pprint
import random
import argparse
from src.model import Tree
from src.utils import loadData, accuracy, split

ITERS = 10
SPLIT_RATIO = 0.8
PATH = './data/final_dataset.csv'

data, target, attr = loadData(path=PATH)
seeds = random.sample([i for i in range])
model = Tree(attr)
model.growTree(data, max_d=14)
# print(attr)
# pprint.pprint(model.tree, width=1)

model.printTree()
score = accuracy(data, model)
print(score)
