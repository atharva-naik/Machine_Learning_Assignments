# Assignment-1: Decision trees
Decision tree for stimulus test type classification for BCI (Brain Computer Interface) data, from electrode variances. 

### Setting up:

```bash
pip install -r requirements.txt
```

### How to run:

#### For training decision tree:
with terminal arguments
```bash
python train.py --num_splits (number_of_fold_for_cross_validation) --save --path (filepath_for_dataset) --thresh (maximum_number_of_outliers_features_allowed_in_a_datapoint) --iters (max_number_of_removals_for_outlier_removing_function)
```
all arguments have been defaulted as well, so directly using the command below is enough
```bash
python train.py 
```
