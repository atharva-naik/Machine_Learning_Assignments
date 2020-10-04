# Assignment-1: Decision trees
Decision tree for stimulus test type classification for BCI (Brain Computer Interface) data, from electrode variances. 

### Setting up:

```bash
pip install -r requirements.txt
```

### Prepare dataset:

#### Preparing the file
Prepare a single file with fields: ['filename', 'electrode 0', 'electrode 1', ... , 'electrode 13', 'test-type'] where test-type is out of ['FIVE BOX 1', 'FIVE BOX 2', 'FIVE BOX 3', 'IMAGE SEARCH', 'HAND SHAKE'] and is the class to be predicted. The values within each row are variances of the elcetrode values in each file. github link to download the entire code and the dataset is:
https://github.com/atharva-naik/Machine_Learning_Assignments.git  (the folder is named 14_ML_A1)

### How to run:

#### For training decision tree:
```bash
python train.py --iters (number_of_splits_to_calculate_accuracu) --test_size (ratio_of_test_to_total) --val_size (ratio_of_val_to_total) --path (filepath_for_dataset) --upper (upper_depth_search_bound) --lower (lower_depth_search_bound)
```

