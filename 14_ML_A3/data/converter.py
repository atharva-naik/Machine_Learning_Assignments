import os
import argparse
import warnings
import pandas as pd 

## terminal arguments ##
parser = argparse.ArgumentParser()
parser.add_argument("-p","--path", type=str, default='input.arff', help="path to arff file")
parser.add_argument("-o","--op_path", type=str, default='output.csv', help="path to csv output")
parser.add_argument("-t","--as_tsv", action="store_true", default=False)
parser.add_argument("-v","--verbose", action="store_true", default=False)
args = parser.parse_args()

f = open(args.path,"r")
attributes = []
for line in f :
    if len(line) > 1 and line.split()[0].startswith("@attribute") :
        attr_name = ' '.join(line.split()[1:])
        attributes.append(attr_name)
f.close()

if args.verbose:
    print("The attributes are :\n")
    i = 0
    for attr_name in attributes :
        i += 1
        print(attr_name, end = ' ')
        if i%5 == 0 :
            print()

if i%5 == 0 :
    print()

df, flag = [], False
f = open(args.path,"r")
for line in f :
    if flag :
        temp = {}
        for i, attr_name in enumerate(attributes) :
            temp[attr_name] = line.split(',')[i].strip()
        df.append(temp) 
    if line.startswith("@data") :
        flag = True
        continue

df = pd.DataFrame(df)
print(f"{len(df)} lines were read ...")
if args.as_tsv :
    if args.op_path.split('.')[-1] != "tsv" :
        raise(Exception("Please use tsv as a file extension"))
    df.to_csv(args.op_path, sep='\t', index=False)
    print(f"saved at {os.getcwd()}/{args.op_path} ...")
else :
    if args.op_path.split('.')[-1] != "csv" :
        raise(Exception("Please use csv as a file extension"))
    df.to_csv(args.op_path, index=False)
    print(f"saved at {os.getcwd()}/{args.op_path} ...")