import os
import pprint
import argparse
import subprocess
import seaborn as sns
from subprocess import call
import matplotlib.pyplot as plt
# from multiprocessing import Pool

RUN_MODELS = False
PLOT_GRAPHS = True
RESULTS = "results.txt"
SEED = 666
RESULTS_ROOT = "results"
LEARNING_RATES = [0.1,0.01,0.001,0.0001,0.00001]
ACTIVATIONS = ['relu','tanh','logistic']
ARCHITECTURES = [[],[2],[6],[2,3],[3,2]]
OPTIMIZERS = ['adam','sgd']

process = "python mlp.py --w {} --hidden_list \"{}\" --lr {} --optim {} --seed {} --activation {}"
process_perceptron = "python mlp.py --w {} --lr {} --seed {}"
def run(args, process) :
    print(process.format(*args))
    process = process.split()
    i = 0
    for j, item in enumerate(process) :
        if "{}" in item :
            process[j] = item.format(args[i])
            i += 1
    # print(process)   
    subprocess.call(process, stdout=subprocess.PIPE)

try :
    print(f"storing plots at {RESULTS_ROOT}")
    os.mkdir(RESULTS_ROOT)
except :
    print(f"{RESULTS_ROOT} already exists")

# to clear results of previous runs
if RUN_MODELS :
    f = open(RESULTS,"w")
    f.close()
    f = open(RESULTS,"a")

    for lr in LEARNING_RATES :
        for hidden_layers in ARCHITECTURES :
            if len(hidden_layers) == 0 :
                args_list = (RESULTS, lr, SEED)
                run(args_list, process_perceptron)      
            else :
                for optim in OPTIMIZERS :
                    for activation in ACTIVATIONS :
                        # print("running learning rate="+str(lr)+" architecture="+str(hidden_layers)+" optimizer="+optim+" activation="+activation+" seed="+str(SEED)+" ...")
                        args_list = (RESULTS, hidden_layers, lr, optim, SEED, activation)
                        run(args_list, process)

    f.close()

def plot(x, y, title, xlbl, ylbl, output) :
    plt.title(title)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.xticks([i for i in range(len(x))], labels=x)
    plt.yticks(y, labels=[round(i,3) for i in y])
    plt.plot([i for i in range(len(x))], y)
    plt.savefig(output)
    plt.clf()

if PLOT_GRAPHS == True :
    f = open(RESULTS, "r")

    accus = []
    models = []
    for line in f :
        models.append(' '.join(line.split()[:-1]))
        accus.append(line.split()[-1])
    print(f"max accuracy={max(accus)}")
    best_model = models[accus.index(max(accus))]
    print(f"best model is:\n{best_model}")
    activation = best_model.split()[-2].split('=')[-1] 
    optimizer = best_model.split()[-3].split('=')[-1]
    print(f"picking optimizer={optimizer} and activation={activation}")

    model_lr_accu = {}
    lr_model_accu = {}
    for i, line in enumerate(models) :
        if ((activation in line) and (optimizer in line)) or ('[]' in line) :
            model = str(line.split('=')[2]).split(']')[0]+']'
            lr = line.split('=')[1].split()[0]
            if model not in model_lr_accu.keys() :
                model_lr_accu[model] = {}
                model_lr_accu[model][float(lr)] = float(accus[i].split('=')[-1])
            else :
                model_lr_accu[model][float(lr)] = float(accus[i].split('=')[-1])
            
            if lr not in lr_model_accu.keys() :
                lr_model_accu[float(lr)] = {}
                lr_model_accu[float(lr)][model] = float(accus[i].split('=')[-1])
            else :
                lr_model_accu[float(lr)][model] = float(accus[i].split('=')[-1])
    
    for i, line in enumerate(models) :
        if ((activation in line) and (optimizer in line)) or ('[]' in line) :
            model = str(line.split('=')[2]).split(']')[0]+']'
            lr = line.split('=')[1].split()[0]            
            lr_model_accu[float(lr)][model] = float(accus[i].split('=')[-1])

    pprint.pprint(model_lr_accu, width=1)
    pprint.pprint(lr_model_accu, width=1)

    for model in model_lr_accu :
        op_name = 'model='+model+'.png'
        print(f"saving at {RESULTS_ROOT+'/'+op_name} ...") 
        plot(list(model_lr_accu[model].keys()), 
            list(model_lr_accu[model].values()), 
            'accuracy of '+model, 
            'learning rate (log scale)', 
            'accuracy',
            RESULTS_ROOT+'/'+op_name)

    for lr in lr_model_accu :
        op_name = 'lr='+str(lr)+'.png'
        print(f"saving at {RESULTS_ROOT+'/'+op_name} ...") 
        plot(list(lr_model_accu[lr].keys()), 
            list(lr_model_accu[lr].values()), 
            'accuracy of model for lr='+str(lr), 
            'model architecture', 
            'accuracy',
            RESULTS_ROOT+'/'+op_name)
    