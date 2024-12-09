import sys
import os
import autograd.numpy as np
from glm_hmm_utils import load_cluster_arr, load_session_fold_lookup, \
    load_data, create_violation_mask, launch_glm_hmm_job, \
    update_features
from glm_hmm_utils import load_glm_vectors, load_global_params, fit_glm_hmm # my addition
import autograd.numpy.random as npr
import json

D = 1  # data (observations) dimension
C = 2  # number of output types/categories
N_em_iters = 300  # number of EM iterations

K_vals = [2] #[2, 3, 4, 5] # number of states 
num_folds = 5
N_initializations = 20

USE_CLUSTER = False

train_test_split = False # change this flag if you want to split train/test here

if __name__ == '__main__':
    data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/'
    results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/global_fit/'

    with open(data_dir + 'labels_for_plot.json', 'r') as f:
        labels_for_plot = json.load(f)
    print(labels_for_plot)

    if USE_CLUSTER:
        z = int(sys.argv[1])
    else:
        z = 0 

    for group in range(1,4): # iterate through groups 1-3 
        group_str = f'{group:02d}'
        print(f"For group {group}:")

        num_folds = 5
        global_fit = True
        # perform mle => set transition_alpha to 1
        transition_alpha = 1
        prior_sigma = 1

        cluster_arr = []
        for K in K_vals:
            for i in range(num_folds):
                for j in range(N_initializations):
                    cluster_arr.append([K, i, j])
        [K, fold, iter] = cluster_arr[z]
        print(f'K: {K}, fold: {fold}, iter: {iter}')
        # print(f'cluster_arr: {cluster_arr}')

        # mine for testing
        # K = 2
        # fold = 0
        # iter = 0
        
        #  read in data and train/test split
        subj_file = data_dir + group_str + '_all_subj_concat.npz'
        trial_fold_lookup_table = load_session_fold_lookup(data_dir + group_str + '_all_subj_concat_trial_fold_lookup.npz')

        # inpt, y = load_data(subj_file)
        container = np.load(subj_file, allow_pickle=True)
        data = [container[key] for key in container]
        inpt = data[0]
        y = data[1]

        # remove features if needed
        # inpt = inpt[:, feat_idxs_to_keep]
        # print(np.shape(inpt))

        #  append a column of ones to input to represent the bias covariate:
        inpt = np.hstack((inpt, np.ones((len(inpt),1))))
        y = y.astype('int')
        # # Identify violations for exclusion:
        violation_idx = np.where(y == -1)[0]
        nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                    inpt.shape[0])

        #  GLM weights to use to initialize GLM-HMM
        init_param_file = results_dir + '/GLM/' + group_str + '_fold_' + str(
            fold) + '/variables_of_interest_iter_0.npz'

        # create save directory for this initialization/fold combination:
        save_directory = results_dir + '/GLM_HMM_K_' + str(
            K) + '/' + group_str + '_fold_' + str(fold) + '/' + '/iter_' + str(iter) + '/'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # launch_glm_hmm_job(): 
        print("Starting inference with K = " + str(K) + "; Fold = " + str(fold) +
            "; Iter = " + str(iter))
        sys.stdout.flush()
        # looks like they are including violation trials bc the mask is supposed to exclude them anyway?
        if train_test_split:
            trials_to_keep = np.where(trial_fold_lookup_table[:,1] == "train")[0] # indices in lookup table that correspond to "train"
            idx_this_fold = [id for id in trials_to_keep] # not really any point in doing this but we'll keep it for readability
    
            # idx_no_viol = np.where(y[:,0] != -1) # exclude any violation trials
            this_inpt, this_y = inpt[idx_this_fold], y[idx_this_fold] 
            this_mask = mask[idx_this_fold]
        else:
            this_inpt, this_y, this_mask = inpt, y, mask
        # Only do this so that errors are avoided - these y values will not
        # actually be used for anything (due to violation mask)
        this_y[np.where(this_y == -1), :] = 1
        # we don't need to do partition_data_by_session since we already partitioned by trial and got the separate variables
        # Read in GLM fit if global_fit = True:
        if global_fit == True:
            _, params_for_initialization = load_glm_vectors(init_param_file)
        else:
            params_for_initialization = load_global_params(init_param_file)
        M = this_inpt.shape[1]
        npr.seed(iter)
        fit_glm_hmm(this_y,
                    this_inpt,
                    this_mask,
                    K,
                    D,
                    M,
                    C,
                    N_em_iters,
                    transition_alpha,
                    prior_sigma,
                    global_fit,
                    params_for_initialization,
                    save_title=save_directory + 'glm_hmm_raw_parameters_itr_' +
                            str(iter) + '.npz')
