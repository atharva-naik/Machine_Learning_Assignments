import argparse
import warnings
from src.utils import loadData, getAccuracy, removeOutliers
from src.model import nBGClassifier, evalKFold

# warnings.filterwarnings(action="ignore", category=UserWarning)
# warnings.filterwarnings(action="ignore", category=SettingWithCopyWarning)
target_field = 'death'
encode = ['location', 'country', 'gender']
fill_method = ['mode', 'mode', 'mode', 'mean', 'mode', 'mode']
fillin = ['location', 'country', 'gender', 'age', 'visiting Wuhan', 'from Wuhan']
drop = ['id', 'reporting date', 'symptom_onset', 'If_onset_approximated', 'hosp_visit_date', 'exposure_start', 'exposure_end', 'recovered']

feature_matrix, features, target = loadData(path='data/Train_D.csv', 
                                            fillin=fillin, 
                                            fill_method=fill_method, 
                                            encode=encode, 
                                            drop=drop, 
                                            clean=True, 
                                            save=True, 
                                            target_field=target_field)
# print(features)
print(feature_matrix.shape)
# print(feature_matrix)
print(target.shape)
print(target)
classes = sorted(list(set(target)))
model = nBGClassifier(classes=[0,1])
# model.fit(feature_matrix, target)
# preds = model.classify(feature_matrix)
# print(set(preds), set(target))
evalKFold(model, feature_matrix, target)
feature_matrix = removeOutliers(feature_matrix, thresh=2, verbose=True)
evalKFold(model, feature_matrix, target)
# print(model.class_dist)
# print(model.means)
# print(getAccuracy(preds, target))


