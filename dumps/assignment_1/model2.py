#!/usr/bin/env python3
import copy
import random
import numpy as np
from .utils import gini_index

class Node:
    def __init__(self, n_samples, class_dist, pred_class, gini_idx):
        self.n_samples = n_samples
        self.class_dist = class_dist 
        self.pred_class = pred_class
        self.gini_idx = gini_idx
        self.attr = ''
        self.thresh = 0
        self.left = None
        self.rtee = None

class DTClassifier:
    def __init__(self, attr=[], depth=0):
        self.max_depth = depth
        self.attr = attr
        self.tree = None

    def growTree(self, data, labels, max_d=3):
        classes_present = sorted(list(set(labels)))
        class_dist = [labels.count(x) for x in classes_present]
        pred_class = classes_present[np.argmax(class_dist)]
        n_samples = sum(class_dist)
        gini_idx(gini_index)
        node = Node(n_samples, class_dist, pred_class, gini_idx)

    def cprint(self, text, r=255, g=0, b=0):
        print("\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text))

    def traverse(self, sample, trace=0): 
        node = self.tree
        while node.left:
            if sample[node.attr] <= node.thresh:
                if trace:
                    self.cprint('{} <= {}: as val[{}]={}'.format(node.attr, node.thresh, node.attr, sample[node.attr]), r=255, g=255)
                node = node.left
            else:
                if trace:
                    self.cprint('{} > {}: as val[{}]={}'.format(node.attr, node.thresh, node.attr, sample[node.attr]), r=255, g=255)
                node = node.right

        return node.pred_class

    def predict(self, samples, trace=0):
        predictions = []
        for sample in samples:
            predictions.append(self.traverse(sample, trace))

        return predictions

    def printTreeRec(self, tree, d):
        tabs = ''
        for i in range(d): tabs += '    '
        if type(tree) == tuple:
            self.cprint(tabs + str(tree[1]) + ' "' + tree[0] + '" out of ' + str(tree[2]))
        else:
            keys = list(tree.keys())
            print(tabs + 'L{}: '.format(d) + keys[0])
            self.printTreeRec(tree[keys[0]], d+1)
            print(tabs + 'R{}: '.format(d) + keys[1])
            self.printTreeRec(tree[keys[1]], d+1)

    def printTree(self):
        tree = self.tree
        print("begin")
        self.printTreeRec(tree, 1)


