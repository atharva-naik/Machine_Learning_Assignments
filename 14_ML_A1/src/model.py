#!/usr/bin/env python3
import copy
import math
import pydot
import random
import numpy as np
from .utils import gini_index, accuracy

def truncate(number, digits=4) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def draw(tree, parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    tree.add_edge(edge)

def cprint(text, r=255, g=0, b=0):
    print("\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text))

def visit(model, tree, node, count, parent=None):
    for k,v in node.items():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if k == 'root': # handle case of root node
                if parent:
                    draw(tree, parent, k)
                visit(model, tree, v, count, k)
            else: # non leaf node
                u = k.split()
                z = list(v.keys())[0].split()
                check_val = z[0] + " " + z[1] + " " + z[3]
                if model.stops.__contains__(check_val):
                    draw(tree, parent, 'e'+u[1]+' '+u[2]+' '+u[3][:6])
                    class_types = ['FIVE BOX 1', 'FIVE BOX 2', 'FIVE BOX 3', 'IMAGE SEARCH', 'HAND SHAKE']
                    dist = model.dists[check_val]
                    pred = class_types[dist.index(max(dist))]
                    v = (pred, sum(dist), max(dist), dist)
                    if not(count.__contains__(v[0])):
                        count[v[0]] = 1
                    else:
                        # print(v[0])
                        count[v[0]] += 1
                    draw(tree, 'e'+u[1]+' '+u[2]+' '+u[3][:6], v[0]+':'+str(count[v[0]]))
                else:
                    if parent:    
                        draw(tree, parent, 'e'+u[1]+' '+u[2]+' '+u[3][:6])
                    visit(model, tree, v, count, 'e'+u[1]+' '+u[2]+' '+u[3][:6])
        else: # leaf node (node just above the predicted class value)
            u = k.split()
            draw(tree, parent, 'e'+u[1]+' '+u[2]+' '+u[3][:6])
            # drawing the label using a distinct name
            if not(count.__contains__(v[0])):
                count[v[0]] = 1
            else:
                count[v[0]] += 1
            draw(tree, 'e'+u[1]+' '+u[2]+' '+u[3][:6], v[0]+':'+str(count[v[0]]))

class Tree:
    def __init__(self, attr=[]):
        self.tree = {}   # tree represented as dictionary
        self.attr = attr # attributes
        self.dists = {} # class distributions at each node
        # node values (threshold and attribute) which when encountered should directly predict by node distribution instead of traversing the tree
        # it is utilised here to implement pruning
        self.stops = {}  
        self.depth = 0
    # growTree grows the tree recursively from the attributes upto the max depth
    def growTree(self, data, max_d = 3):
        # build decision tree
        chosen_attr = ''
        classes_present = [item['test-type'] for item in data]
        class_types = ['FIVE BOX 1', 'FIVE BOX 2', 'FIVE BOX 3', 'IMAGE SEARCH', 'HAND SHAKE']
        class_dist = [classes_present.count(x) for x in class_types]
        min_G = gini_index(class_dist)
        # print("1:{} {}".format(min_G, class_dist))
        if max_d == 0: 
            final_class = class_types[class_dist.index(max(class_dist))]
            return (final_class, classes_present.count(final_class), len(classes_present), class_dist)

        if len(set(classes_present)) == 1:
            return (classes_present[0], classes_present.count(classes_present[0]), len(classes_present), class_dist)
        
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
        self.dists[chosen_attr+' '+str(thresh)] = class_dist

        return tree

    def traverse(self, tree, val, trace=0): 
        if type(tree) == tuple:
            return tree

        keys = list(tree.keys())
        attr, thresh = keys[0].split('<=')
        attr, thresh = attr.strip(), float(thresh)
        if self.stops.__contains__(attr + ' ' + str(thresh)):
            class_types = ['FIVE BOX 1', 'FIVE BOX 2', 'FIVE BOX 3', 'IMAGE SEARCH', 'HAND SHAKE']
            dist = self.dists[attr + ' ' + str(thresh)]
            pred = class_types[dist.index(max(dist))]
            return (pred, max(dist), sum(dist), dist)

        if val[attr] <= thresh:
            if trace: 
                cprint(keys[0]+': as val[{}]={}'.format(attr, val[attr]), r=255, g=255)
            pred = self.traverse(tree[keys[0]], val, trace)
        else:
            if trace: 
                cprint(keys[1]+': as val[{}]={}'.format(attr, val[attr]), r=255, g=255)
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
            cprint(tabs + str(tree[1]) + ' "' + tree[0] + '" in ' + str(tree[3]) + ' tot='+str(tree[2]))
        else:
            keys = list(tree.keys())
            print(tabs + 'L{}: '.format(d) + keys[0])
            self.printTreeRec(tree[keys[0]], d+1)
            print(tabs + 'R{}: '.format(d) + keys[1])
            self.printTreeRec(tree[keys[1]], d+1)

    def printTreeTerminal(self):
        cprint("\nPRINTING TREE IN TERMINAL ...\n",g=180)
        tree = self.tree
        print("begin")
        self.printTreeRec(tree, 1)


    def printTreeImage(self, name='decision_tree'):
        graph = pydot.Dot(graph_type='graph')
        cprint("plot saved as {}.png".format(name),r=151,b=38,g=189)
        tree = {}
        tree['root'] = self.tree
        visit(self, graph, tree, {})  #pass the dict here in the 2nd argument
        graph.write_png(name+'.png')

    def recPrune(self, curr, data):
        try:
            keys = list(curr.keys())
        except AttributeError:
            return 
        if isinstance(curr[keys[0]], tuple) and isinstance(curr[keys[1]], tuple):
            class_types = ['FIVE BOX 1', 'FIVE BOX 2', 'FIVE BOX 3', 'IMAGE SEARCH', 'HAND SHAKE']
            acc1 = accuracy(data, self) 
            
            try:
                attr, thresh = keys[0].split('<=')
            except:
                attr, thresh = keys[0].split('>')
            
            attr = attr.strip()
            thresh = thresh.strip()
            self.stops[attr + ' ' + thresh] = 1
            acc2 = accuracy(data, self) 
            # print((attr, thresh, acc1, acc2))

            if acc2 < acc1:
                self.stops.pop(attr + ' ' + thresh, None)
            else:
                print("node with attribute={} and threshold={} collapsed".format(attr, thresh[:6]))

        else:
            self.recPrune(curr[keys[0]], data)
            self.recPrune(curr[keys[1]], data)

    def postPrune(self, val_data):
        cprint("\nBEGINING PRUNING ...\n",r = 0,b=255,g=255)
        self.recPrune(self.tree, val_data)
