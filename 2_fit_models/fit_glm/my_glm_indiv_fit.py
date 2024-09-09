# Fit GLM to each IBL animal separately
import autograd.numpy as np
import autograd.numpy.random as npr
import os
from glm_utils import load_session_fold_lookup, load_data, load_subj_list, \
    fit_glm, plot_input_vectors, append_zeros
from tqdm import tqdm # my addition
import matplotlib.pyplot as plt

C = 2  # number of output types/categories
N_initializations = 10
npr.seed(65)

all_labels = ['stim_probe X', 'stim_probe Y', 'stim_probe dist', 'stim_probe angle',
                'stim_1 X', 'stim_1 Y', 'stim_1 dist', 'stim_1 angle',
                'stim_2 X', 'stim_2 Y', 'stim_2 dist', 'stim_2 angle',
                'stim_3 X', 'stim_3 Y', 'stim_3 dist', 'stim_3 angle',
                'prev_resp', 'prev_acc', 'bias']

# doing_feature_selection = True # change this flag if you are using this code to do feature selection or not

# for manual feature selection
# features_to_remove = ['stim_probe X', 'stim_probe Y', 'stim_1 X', 'stim_1 Y', 
#                       'stim_2 X', 'stim_2 Y', 'stim_3 X', 'stim_3 Y', 'bias']  # Update this list with features you want to remove

# # # Update features and labels based on removal
# feat_idxs_to_keep = update_features(features_to_remove, all_labels)
# labels_for_plot = [all_labels[i] for i in feat_idxs_to_keep]
# print(labels_for_plot)

# if 'bias' not in features_to_remove:
#     feat_idxs_to_keep = feat_idxs_to_keep[:,-1] # remove last term so it doesn't cause an issue with input
labels_for_plot = all_labels # temp

if __name__ == '__main__':
    data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/data_by_subj/'
    num_folds = 5

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

            for fold in range(num_folds):
                this_results_dir = results_dir + group_str + '_' + subj + '/'

                # Load data
                container = np.load(subj_file, allow_pickle=True)
                data = [container[key] for key in container]
                inpt = data[0]
                y = data[1]
                y = y.astype('int')

                figure_directory = this_results_dir + "GLM/fold_" + str(fold) + '/'
                if not os.path.exists(figure_directory):
                    os.makedirs(figure_directory)

                # Subset to sessions of interest for fold
                # sessions_to_keep = session_fold_lookup_table[np.where(
                #     session_fold_lookup_table[:, 1] != fold), 0]
                # idx_this_fold = [
                #     str(sess) in sessions_to_keep and y[id, 0] != -1
                #     for id, sess in enumerate(session)
                # ]
                # this_inpt, this_y, this_session = inpt[idx_this_fold, :], \
                #                                 y[idx_this_fold, :], \
                                                # session[idx_this_fold]

                idx_no_viol = np.where(y[:,0] != -1) # exclude any violation trials
                this_inpt, this_y = inpt[idx_no_viol], y[idx_no_viol] # exclude any violation trials
                
                assert len(
                    np.unique(this_y)
                ) == 2, "choice vector should only include 2 possible values"
                train_size = this_inpt.shape[0]

                M = this_inpt.shape[1]
                loglikelihood_train_vector = []

                for iter in range(N_initializations):
                    loglikelihood_train, recovered_weights = fit_glm([this_inpt],
                                                                    [this_y], M, C)
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
