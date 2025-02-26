# Fit GLM to each IBL animal separately
import autograd.numpy as np
import autograd.numpy.random as npr
import os
from glm_utils import load_session_fold_lookup, load_data, load_subj_list, \
    fit_glm, plot_input_vectors, append_zeros
from tqdm import tqdm # my addition
import matplotlib.pyplot as plt
import json

C = 2  # number of output types/categories
N_initializations = 10
npr.seed(65)

train_test_split = True # change this flag if you want to split train/test here

if __name__ == '__main__':
    global_data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/'
    data_dir = global_data_dir + 'data_by_subj/'
    num_folds = 5

    with open(global_data_dir + 'labels_for_plot.json', 'r') as f:
        labels_for_plot = json.load(f)
    print(labels_for_plot)

    results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/individual_fit/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for group in range(1,4):
        group_str = f'{group:02d}' # which group we are currently looking at 

        subj_list = load_subj_list(data_dir + group_str + '_final_subject_list.npz')

        for subj in tqdm(subj_list, desc=f'Group {group}'):
            # Fit GLM to data from single animal:
            subj_file = data_dir + group_str + '_' + subj + '_processed.npz'
            # session_fold_lookup_table = load_session_fold_lookup(
            #     data_dir + subj + '_session_fold_lookup.npz')
            trial_fold_lookup_table = load_session_fold_lookup(global_data_dir + 'data_by_subj/' + group_str + '_' + subj + '_trial_fold_lookup.npz')

            for fold in range(num_folds):
                this_results_dir = results_dir + group_str + '_' + subj + '/'

                # Load data
                # container = np.load(subj_file, allow_pickle=True)
                # data = [container[key] for key in container]
                # inpt = data[0]
                # y = data[1]
                inpt, y = load_data(subj_file)
                y = y.astype('int')

                figure_directory = this_results_dir + "GLM/fold_" + str(fold) + '/'
                if not os.path.exists(figure_directory):
                    os.makedirs(figure_directory)

                idx_no_viol = np.where(y[:,0] != -1) # exclude any violation trials
                if train_test_split: 
                    trials_to_keep = np.where(trial_fold_lookup_table[:,1] == "train")[0] # indices in lookup table that correspond to "train"
                    idx_this_fold = [id for id in trials_to_keep if y[id,0] != -1]

                    this_inpt, this_y = inpt[idx_this_fold], y[idx_this_fold] 
                else:
                    this_inpt, this_y = inpt[idx_no_viol], y[idx_no_viol] 
                assert len(np.unique(this_y)) == 2, "choice vector should only include 2 possible values"
                # train_size = this_inpt.shape[0]

                M = this_inpt.shape[1]
                loglikelihood_train_vector = []

                for iter in range(N_initializations):
                    loglikelihood_train, recovered_weights = fit_glm([this_inpt],
                                                                    [this_y], 
                                                                    None, # no tags
                                                                    M, C)
                    weights_for_plotting = append_zeros(recovered_weights)
                    plot_input_vectors(weights_for_plotting,
                                    figure_directory,
                                    title="GLM fit; Final LL = " +
                                    str(loglikelihood_train),
                                    save_title='init' + str(iter),
                                    labels_for_plot=labels_for_plot)
                    loglikelihood_train_vector.append(loglikelihood_train)
                    np.savez(
                        figure_directory + 'variables_of_interest_iter_' +
                        str(iter) + '.npz', loglikelihood_train, recovered_weights)
                plt.close('all') # close all figures after each group is done to save memory
