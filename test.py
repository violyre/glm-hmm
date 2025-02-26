import numpy as np

subj_lookup = "C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/data_by_subj/01_00002_trial_fold_lookup.npz"
container = np.load(subj_lookup, allow_pickle=True)
data = [container[key] for key in container]
session_fold_lookup_table = data[0]

print(session_fold_lookup_table)