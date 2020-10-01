#!/usr/bin/env python3
import copy
import random
import numpy as np
from .utils import gini_index

class Tree:
    def __init__(self, attr=[]):
        self.tree = {}   # tree represented as dictionary
        self.attr = attr # attributes
        self.depth = 0
    # growTree grows the tree recursively from the attributes upto the max depth
    def growTree(self, data, max_d = 3):
        # build decision tree
        chosen_attr = ''
        classes_present = [item['test-type'] for item in data]
        class_dist = [classes_present.count(x) for x in set(classes_present)]
        min_G = gini_index(class_dist)
        # print("1:{} {}".format(min_G, class_dist))
        if max_d == 0: 
            final_class = max(set(classes_present), key=classes_present.count)
            return (final_class, classes_present.count(final_class), len(classes_present))

        if len(set(classes_present)) == 1:
            return (classes_present[0], classes_present.count(classes_present[0]), len(classes_present))
        
        thresh = 0
        # left_data, right_data = [], []
        for attribute in self.attr:
            # skip non integral attributes
            if not(attribute.startswith('electrode')):
                continue

            data = sorted(data, key=lambda item: item[attribute])
            tmp_l, tmp_r = data[0:1], data[1:]
            temp = 1000
            
            for i in range(len(data)-1):
                l_classes_present = [item['test-type'] for item in tmp_l]
                r_classes_present = [item['test-type'] for item in tmp_r]
                g_left = gini_index([l_classes_present.count(x) for x in set(l_classes_present)])
                g_right = gini_index([r_classes_present.count(x) for x in set(r_classes_present)])
                
                if temp > (len(tmp_l)*g_left + len(tmp_r)*g_right)/(len(data)):
                    temp = (len(tmp_l)*g_left + len(tmp_r)*g_right)/(len(data))
                    local_thresh = (data[i][attribute] + data[i+1][attribute])*0.5
                    # if class_dist == [6, 9, 4, 3, 1]: print(thresh)
                tmp_l += [data[i+1]] 
                tmp_r = tmp_r[1:]
            
            if temp < min_G:
                chosen_attr = attribute
                thresh = local_thresh
                min_G = temp

        tree = {}
        left_data, right_data = [], [] 
        data = sorted(data, key= lambda x: x[chosen_attr])
        
        for datapoint in data:
            if datapoint[chosen_attr] <= thresh:
                left_data.append(datapoint)
            else:
                right_data.append(datapoint)

        tree[chosen_attr + " <= {}".format(thresh)] = self.growTree(left_data, max_d=max_d-1)
        tree[chosen_attr + " > {}".format(thresh)] = self.growTree(right_data, max_d=max_d-1)
        self.depth = max_d
        self.tree = tree
        
        return tree

    def cprint(self, text, r=255, g=0, b=0):
        print("\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text))

    def traverse(self, tree, val, trace=0): 
        if type(tree) == tuple:
            return tree

        keys = list(tree.keys())
        attr, thresh = keys[0].split('<=')
        attr, thresh = attr.strip(), float(thresh)
        
        if val[attr] <= thresh:
            if trace: 
                self.cprint(keys[0]+': as val[{}]={}'.format(attr, val[attr]), r=255, g=255)
            pred = self.traverse(tree[keys[0]], val, trace)
        else:
            if trace: 
                self.cprint(keys[1]+': as val[{}]={}'.format(attr, val[attr]), r=255, g=255)
            pred = self.traverse(tree[keys[1]], val, trace)
        
        return pred

    def predict(self, samples, trace=0):
        predictions = []
        for sample in samples:
            predictions.append(self.traverse(self.tree, sample, trace))
        
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


