# Assignment-1: Decision trees
Decision tree for stimulus test type classification for BCI (Brain Computer Interface) data, from electrode variances. 

### Setting up:

```bash
pip install -r requirements.txt
```

### How to run:

#### For training decision tree:
```bash
python train.py --iters (number_of_splits_to_calculate_accuracu) --test_size (ratio_of_test_to_total) --val_size (ratio_of_val_to_total) --path (filepath_for_dataset) --upper (upper_depth_search_bound) --lower (lower_depth_search_bound)
```