import autograd.numpy as np
import autograd.numpy.random as npr
import os
from glm_utils import load_session_fold_lookup, load_data, fit_glm, \
    plot_input_vectors, append_zeros, \
    plot_feature_selection_ll, \
    update_features
from tqdm import tqdm # my addition
import statistics # for variance

results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/global_fit/'

all_labels = ['stim_probe X', 'stim_probe Y', 'stim_probe dist', 'stim_probe angle',
                'stim_1 X', 'stim_1 Y', 'stim_1 dist', 'stim_1 angle',
                'stim_2 X', 'stim_2 Y', 'stim_2 dist', 'stim_2 angle',
                'stim_3 X', 'stim_3 Y', 'stim_3 dist', 'stim_3 angle',
                'prev_resp', 'prev_acc', 'bias']

num_folds = 5

container = np.load(results_dir + 'GLM/' + 'feat_select_all_lls.npz', allow_pickle=True)
data = [container[key] for key in container]
ll_vectors_allgroups = data[0]
original_ll_allgroups = data[1]
plot_feature_selection_ll(all_labels[:-1],ll_vectors_allgroups,original_ll_allgroups,num_folds,
                            directory=results_dir + 'GLM/',
                            title=f'All Groups, Averaged Across Folds',
                            save_title="feat_select_ll_allgroups_diff.png",
                            type='GroupDiff')

#%% see how many subjects are being kept in group 1
import autograd.numpy as np
import json

path = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/data_by_subj/'
final_subject_list_all = {}
for group in range(1,4):
    group_str = f'{group:02d}'
    container = np.load(path + group_str + '_final_subject_list.npz', allow_pickle = True)
    data = [container[key] for key in container]
    slist = data[0]
    print(f"group {group}: {np.shape(slist)}")

    final_subject_list_all[group] = [subj for subj in slist]
print(final_subject_list_all)
# json = json.dumps(final_subject_list_all)
# json = json.dumps(final_subject_ids_dict)
# f = open(path + 'total_final_subject_list.json', "w")
# f.write(json)
# f.close()

# %%
import pandas as pd 
import os

data_path = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/'
filename = '16_01_00003_02_trialdata.csv'
data = pd.read_csv(os.path.join(data_path, filename))
data = data.drop('Unnamed: 0', axis=1)
data.head()
# %%
