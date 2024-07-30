import sys
import os
import autograd.numpy as np
from glm_hmm_utils import load_cluster_arr, load_session_fold_lookup, \
    load_data, create_violation_mask, launch_glm_hmm_job
from glm_hmm_utils import load_glm_vectors, load_global_params, fit_glm_hmm # my addition
import autograd.numpy.random as npr

D = 1  # data (observations) dimension
C = 2  # number of output types/categories
N_em_iters = 300  # number of EM iterations

K_vals = [2, 3, 4, 5] # number of states 
num_folds = 5
N_initializations = 20

USE_CLUSTER = False

if __name__ == '__main__':
    data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/'
    results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/global_fit/'

    if USE_CLUSTER:
        z = int(sys.argv[1])
    else:
        z = 0 

    for group in range(1,4): # iterate through groups 1-3 
        group_str = f'{group:02d}'

        num_folds = 5
        global_fit = True
        # perform mle => set transition_alpha to 1
        transition_alpha = 1
        prior_sigma = 100

        # cluster_arr = []
        # for K in K_vals:
        #     for i in range(num_folds):
        #         for j in range(N_initializations):
        #             cluster_arr.append([K, i, j])
        # [K, fold, iter] = cluster_arr[z]
        # print(f'K: {K}, fold: {fold}, iter: {iter}')
        # print(f'cluster_arr: {cluster_arr}')
        K = 3
        fold = 0
        iter = 0
        
        #  read in data and train/test split
        subj_file = data_dir + group_str + '_all_subj_concat.npz'
        # session_fold_lookup_table = load_session_fold_lookup(
        #     data_dir + 'all_animals_concat_session_fold_lookup.npz')

        # input, y = load_data(subj_file)
        container = np.load(subj_file, allow_pickle=True)
        data = [container[key] for key in container]
        input = data[0]
        y = data[1]
        #  append a column of ones to inpt to represent the bias covariate:
        input = np.hstack((input, np.ones((len(input),1))))
        y = y.astype('int')
        # # Identify violations for exclusion:
        violation_idx = np.where(y == -1)[0]
        nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                    input.shape[0])

        #  GLM weights to use to initialize GLM-HMM
        init_param_file = results_dir + '/GLM/' + group_str + '_fold_' + str(
            fold) + '/variables_of_interest_iter_0.npz'

        # create save directory for this initialization/fold combination:
        save_directory = results_dir + '/GLM_HMM_K_' + str(
            K) + '/' + 'fold_' + str(fold) + '/' + '/iter_' + str(iter) + '/'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # launch_glm_hmm_job(): 
        print("Starting inference with K = " + str(K) + "; Fold = " + str(fold) +
            "; Iter = " + str(iter))
        sys.stdout.flush()
        idx_no_viol = np.where(y[:,0] != -1) # exclude any violation trials
        this_input, this_y = input[idx_no_viol], y[idx_no_viol] # exclude any violation trials
        this_mask = mask[idx_no_viol]
        # Only do this so that errors are avoided - these y values will not
        # actually be used for anything (due to violation mask)
        this_y[np.where(this_y == -1), :] = 1
        # Read in GLM fit if global_fit = True:
        if global_fit == True:
            _, params_for_initialization = load_glm_vectors(init_param_file)
        else:
            params_for_initialization = load_global_params(init_param_file)
        M = this_input.shape[1]
        npr.seed(iter)
        fit_glm_hmm(this_y,
                    this_input,
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
