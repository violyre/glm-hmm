# modified from 2_create_design_mat.py
import numpy as np
from sklearn import preprocessing
import numpy.random as npr
import os
import json
from collections import defaultdict

npr.seed(65)

if __name__ == '__main__':
    data_dir = "C:\Users\violy\Documents\~PhD\Lab\SC\TCP_data_organized_updated\TCP_data_organized_updated"
    num_folds = 5 #why?
    
    # Create directory for results:
    results_dir = '../../results/vdata_global_fit/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    
    subj_file = data_dir + 'all_subj_concat.npz'