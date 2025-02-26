import sys
import os
import autograd.numpy as np
from glm_hmm_utils import load_session_fold_lookup, \
    load_data, create_violation_mask
from glm_hmm_utils import load_glm_vectors, load_global_params, load_tags # my addition
import autograd.numpy.random as npr
import json
import ssm

D = 1  # data (observations) dimension
C = 2  # number of output types/categories
N_em_iters = 300  # number of EM iterations

K_vals = [2, 3] #[2, 3, 4, 5] # number of states 
num_folds = 1 #5
N_initializations = 1 #20

USE_CLUSTER = False

prior_sigma_values = [0.25, 0.5, 1] #[0.1, 0.5, 1, 2, 5, 10, 25]
prior_alpha_values = [1, 2] #[1, 2, 5, 7, 10]

if __name__ == '__main__':
    data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/'
    results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/global_fit/'

    with open(data_dir + 'labels_for_plot.json', 'r') as f:
        labels_for_plot = json.load(f)
    print(labels_for_plot)

    # cluster_arr = []
    # for K in K_vals:
    #     for i in range(num_folds):
    #         for j in range(N_initializations):
    #             cluster_arr.append([K, i, j])
    # print(f'cluster_arr: {cluster_arr}')

    for K in K_vals:
        fold = 0
    #     # [K, fold, iter] = cluster_arr[z]
    #     # print(f'K: {K}, fold: {fold}, iter: {iter}')

        for group in range(1,4): # iterate through groups 1-3 
            group_str = f'{group:02d}'
            print(f"For group {group}:")

            best_model = None
            best_ll = -np.inf # store best log-likelihood later
            best_sigma = 0
            best_alpha = 0

            # num_folds = 5
            global_fit = True
            # perform mle => set transition_alpha to 1
            # transition_alpha = 1
            # prior_sigma = 1 # originally set as 100
            
            #  read in data and train/test split
            subj_file = data_dir + group_str + '_all_subj_concat.npz'
            tags_file = data_dir + group_str + '_all_subj_tags.npz'
            trial_fold_lookup_table = load_session_fold_lookup(data_dir + group_str + '_all_subj_concat_trial_fold_lookup.npz')

            inpt, y = load_data(subj_file)
            tags = load_tags(tags_file)

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
                K) + '/'
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # launch_glm_hmm_job(): 
            # print("Starting inference with K = " + str(K) + "; Fold = " + str(fold) +
            #     "; Iter = " + str(iter))
            sys.stdout.flush()
            this_inpt, this_y, this_mask = inpt, y, mask
            this_tags = tags
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
            # npr.seed(iter)

            for sigma in prior_sigma_values:
                for alpha in prior_alpha_values:
                    print(f"Trying prior_sigma = {sigma} and transition_alpha = {alpha}")
                    this_hmm = ssm.HMM(K,
                        D,
                        M,
                        observations="input_driven_obs",
                        observation_kwargs=dict(C=C,
                                                prior_sigma=sigma),
                        transitions="sticky",
                        transition_kwargs=dict(alpha=alpha,
                                                kappa=0))
                    this_ll = this_hmm.fit(this_y,
                        inputs=this_inpt,
                        masks=this_mask,
                        tags=this_tags,
                        method="em",
                        num_iters=N_em_iters,
                        initialize=False,
                        tolerance=10 ** -4)
                    # print(this_ll)
                    final_ll = this_ll[-1]
                    if final_ll > best_ll:
                        best_model = this_hmm
                        best_ll = final_ll
                        best_sigma = sigma
                        best_alpha = alpha

            # np.savez(save_directory + 'glm_hmm_best_model.npz',this_hmm.params, this_ll)
            # Save the best model parameters and log-likelihood
            np.savez(save_directory + 'glm_hmm_best_model_' + group_str + '.npz', 
                     params=best_model.params, 
                     log_likelihood=best_ll)

            # Save the best values of ll, sigma, and alpha for each group
            best_values = {
                'best_ll': best_ll,
                'best_sigma': best_sigma,
                'best_alpha': best_alpha
            }
            np.savez(save_directory + 'best_values_' + group_str + '.npz', **best_values)

            print(f"Best prior_sigma for group {group} and K={K}: {best_sigma}")
            print(f"Best transition_alpha for group {group} and K={K}: {best_alpha}")
