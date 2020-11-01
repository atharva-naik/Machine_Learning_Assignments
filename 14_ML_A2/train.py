import argparse
import warnings
from src.utils import loadData, getAccuracy, removeOutliers, cprint
from src.model import nBGClassifier, evalKFold, evalKFoldPCA, seqBackSelect

## terminal arguments ##
parser = argparse.ArgumentParser()
parser.add_argument("-k","--num_splits", type=int, default=5, help="number of folds for cross validation")
parser.add_argument("-i","--iters", type=int, default=101, help="max number of removals for outlier removing function")
parser.add_argument("-p","--path", type=str, default='data/Train_D.csv', help="path to dataset")
parser.add_argument("-t","--thresh", type=int, default=2, help="maximum number of outliers features allowed in a datapoint")
parser.add_argument("-s","--save", action='store_true', default=True, help="save preprocessed dataset")
args = parser.parse_args()

target_field = 'death'
encode = ['location', 'country', 'gender']
fill_method = ['mode', 'mode', 'mode', 'mean', 'mode', 'mode']
fillin = ['location', 'country', 'gender', 'age', 'visiting Wuhan', 'from Wuhan']
drop = ['id', 'reporting date', 'symptom_onset', 'If_onset_approximated', 'hosp_visit_date', 'exposure_start', 'exposure_end', 'recovered']

feature_matrix, features, target = loadData(path=args.path, 
                                            fillin=fillin, 
                                            fill_method=fill_method, 
                                            encode=encode, 
                                            drop=drop, 
                                            clean=True, 
                                            save=args.save, 
                                            target_field=target_field)
# print(features)
# print(feature_matrix.shape)
# print(feature_matrix)
# print(target.shape)
# print(target)

classes = sorted(list(set(target)))
model = nBGClassifier(classes=classes)
# print(set(preds), set(target))

# PART-1
cprint("PART 1: VANILLA NAIVE BAYES WITH 5 FOLD CROSS VALIDATION :", b=255)
evalKFold(model, feature_matrix, target, splits=args.num_splits, verbose=True)

# PART-2
# here the PCA is not benefitting for our case due to the highly skewed dataset, the class wise accuracy on death=1 drops to zero 
cprint("PART 2: PCA & NAIVE BAYES WITH 5 FOLD CROSS VALIDATION :", b=255)
evalKFoldPCA(model, feature_matrix, target, verbose=True)

# PART-3
cprint("PART 3: OUTLIER REMOVAL, SEQUENTIAL BACKWARD SELECTION, NAIVE BAYES WITH 5 FOLD CROSS VALIDATION :", b=255)
# continue removing outliers till max_removed number of entries are removed or till or till number of outlier features becomes less than 1, whichever occurs first
# in our case the accuracy wasn't improving after removing features, so zero features are removed by this function
feature_matrix = removeOutliers(feature_matrix, thresh=args.thresh, verbose=False, max_removed=args.iters)
evalKFold(model, feature_matrix, target, verbose=True)
feature_matrix = seqBackSelect(model, feature_matrix, target)
evalKFold(model, feature_matrix, target, verbose=True)








