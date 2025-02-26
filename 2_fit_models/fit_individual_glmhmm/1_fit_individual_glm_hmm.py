#  Fit GLM-HMM to data from all IBL animals together.  These fits will be
#  used to initialize the models for individual animals
import os
import sys

import autograd.numpy as np
from glm_hmm_utils import load_glm_vectors, load_global_params, fit_glm_hmm # my addition
import autograd.numpy.random as npr

D = 1  # data (observations) dimension
C = 2  # number of output types/categories
N_em_iters = 300  # number of EM iterations

prior_sigma = [2]
transition_alpha = [2]
K_vals = [2, 3] # [2, 3, 4, 5]
num_folds = 5
N_initializations = 2

USE_CLUSTER = False

train_test_split = True # change this flag if you want to split train/test here

if __name__ == '__main__':
    global_data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/'
    data_dir = global_data_dir + 'data_by_subj/'
    results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/individual_fit/'
    sys.path.insert(0, '../fit_global_glmhmm/')

    if USE_CLUSTER:
        z = int(sys.argv[1])
        from glm_hmm_utils import load_cluster_arr, load_session_fold_lookup, \
            load_subj_list, load_data, create_violation_mask, \
            launch_glm_hmm_job
    else:
        z = 0
        from glm_hmm_utils import load_cluster_arr, load_session_fold_lookup, \
            load_subj_list, load_data, create_violation_mask, \
            launch_glm_hmm_job

    num_folds = 1 #5

    cluster_arr = []
    for K in K_vals:
        for i in range(num_folds):
            for j in range(N_initializations):
                for sigma in prior_sigma:
                    for alpha in transition_alpha:
                        cluster_arr.append([sigma, alpha, K, i, j])
    [prior_sigma, transition_alpha, K, fold, iter] = cluster_arr[z]
    print(f'K: {K}, fold: {fold}, iter: {iter}')
    # print(f'cluster_arr: {cluster_arr}')

    iter = int(iter)
    fold = int(fold)
    K = int(K)

    for group in range(1,4):
        group_str = f'{group:02d}' # which group we are currently looking at 

        subj_list = load_subj_list(data_dir + group_str + '_final_subject_list.npz')

        for i, subj in enumerate(subj_list):
            print(f'{group_str}_{subj}')
            subj_file = data_dir + group_str + '_' + subj + '_processed.npz'
            # session_fold_lookup_table = load_session_fold_lookup(
            #     data_dir + subj + '_session_fold_lookup.npz')
            trial_fold_lookup_table = load_session_fold_lookup(global_data_dir + 'data_by_subj/' + group_str + '_' + subj + '_trial_fold_lookup.npz')

            global_fit = False

            # Load data
            container = np.load(subj_file, allow_pickle=True)
            data = [container[key] for key in container]
            inpt = data[0]
            y = data[1]
            y = y.astype('int')

            #  append a column of ones to inpt to represent the bias covariate:
            inpt = np.hstack((inpt, np.ones((len(inpt), 1))))
            y = y.astype('int')

            overall_dir = results_dir + group_str + '_' + subj + '/'

            # Identify violations for exclusion:
            violation_idx = np.where(y == -1)[0]
            nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                        inpt.shape[0])

            init_param_file = global_data_dir + \
                            'best_global_params/best_params_' + group_str + '_K_' + \
                                    str(K) + '.npz'

            # create save directory for this initialization/fold combination:
            save_directory = overall_dir + '/GLM_HMM_K_' + str(
                K) + '/' + 'fold_' + str(fold) + '/' + '/iter_' + str(iter) + '/'
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