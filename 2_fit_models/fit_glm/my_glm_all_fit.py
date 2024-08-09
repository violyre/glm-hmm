#  Fit GLM to all IBL data together

import autograd.numpy as np
import autograd.numpy.random as npr
import os
from glm_utils import load_session_fold_lookup, load_data, fit_glm, \
    plot_input_vectors, append_zeros, \
    feature_selection # my addition
from tqdm import tqdm # my addition

C = 2  # number of output types/categories
N_initializations = 10 # where does this come from?
npr.seed(65)  # set seed in case of randomization

all_labels = ['stim_probe X', 'stim_probe Y', 'stim_probe dist', 'stim_probe angle',
                'stim_1 X', 'stim_1 Y', 'stim_1 dist', 'stim_1 angle',
                'stim_2 X', 'stim_2 Y', 'stim_2 dist', 'stim_2 angle',
                'stim_3 X', 'stim_3 Y', 'stim_3 dist', 'stim_3 angle',
                'prev_resp', 'prev_acc', 'bias']

# Define a function to update the feature list based on features to remove
def update_features(features_to_remove):
    # Filter out the features to remove
    features_to_keep = [label for label in all_labels if label not in features_to_remove]
    return features_to_keep

# Example of features to remove
features_to_remove = ['stim_probe Y', 'stim_1 Y', 'stim_2 Y', 'stim_3 Y']  # Update this list with features you want to remove

# Update features and labels based on removal
labels_for_plot = update_features(features_to_remove)

if __name__ == '__main__':
    data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/'
    num_folds = 5 # why 5 folds?

    # # for use with glm fit for subjects separately 
    # container = np.load(data_dir + 'data_by_subj/subject_list.npz', allow_pickle=True)
    # data = [container[key] for key in container]
    # subject_list = data[0]

    # Create directory for results:
    results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/global_fit/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for group in range(1,2): #range(1,4): # iterate through groups 1-3 
        group_str = f'{group:02d}'

        # Fit GLM to all data
        subj_file = data_dir + group_str + '_all_subj_concat.npz'
        # input, y = load_data(subj_file)
        container = np.load(subj_file, allow_pickle=True)
        data = [container[key] for key in container]
        input = data[0]
        y = data[1]

        print(input)
        # session_fold_lookup_table = load_session_fold_lookup(
        #     data_dir + 'all_animals_concat_session_fold_lookup.npz')

        for fold in range(1,2):#range(num_folds):
            # Subset to relevant covariates for covar set of interest:
            # labels_for_plot = ['stim_probe X', 'stim_probe Y', 'stim_probe dist', 'stim_probe angle',
            #                     'stim_1 X', 'stim_1 Y', 'stim_1 dist', 'stim_1 angle',
            #                     'stim_2 X', 'stim_2 Y', 'stim_2 dist', 'stim_2 angle',
            #                     'stim_3 X', 'stim_3 Y', 'stim_3 dist', 'stim_3 angle',
            #                     'prev_resp', 'prev_acc', 'bias']
            y = y.astype('int')
            print(np.unique(y))
            figure_directory = results_dir + 'GLM/' + group_str + '_fold_' + str(fold) + '/'
            if not os.path.exists(figure_directory):
                os.makedirs(figure_directory)

            # Subset to sessions of interest for fold
            # sessions_to_keep = session_fold_lookup_table[np.where(
            #     session_fold_lookup_table[:, 1] != fold), 0]
            # idx_this_fold = [
            #     str(sess) in sessions_to_keep and y[id, 0] != -1
            #     for id, sess in enumerate(session)
            # ]
            # this_inpt, this_y, this_session = inpt[idx_this_fold, :], y[
            #     idx_this_fold, :], session[idx_this_fold]

            idx_no_viol = np.where(y[:,0] != -1) # exclude any violation trials
            # print(f'idx_this_y: {idx_this_y}')
            this_input, this_y = input[idx_no_viol], y[idx_no_viol] # exclude any violation trials
            # print(f'this_y: {this_y}')
            print(f'shape of y: {np.shape(y)} vs shape of this_y: {np.shape(this_y)}')
            print(f'shape of input: {np.shape(input)} vs shape of this_input: {np.shape(this_input)}')
            
            assert len(
                np.unique(this_y)
            ) == 2, "choice vector should only include 2 possible values"
            train_size = input.shape[0]

            # Adjust the input features based on the labels_for_plot
            feature_indices = [all_labels.index(label) for label in labels_for_plot]
            this_input = this_input[:, feature_indices]  # Select only the relevant columns

            # M = this_inpt.shape[1]
            M = input.shape[1]
            loglikelihood_train_vector = []

            for iter in tqdm(range(N_initializations), desc=f'Group {group_str}, Fold {fold}', unit='init'):  
                # GLM fitting should be
                # independent of initialization, so fitting multiple
                # initializations is a good way to check that everything is
                # working correctly
                loglikelihood_train, recovered_weights, deviance = fit_glm([this_input],
                                                                [this_y], M, C)
                # print(f'iter {iter}, recovered_weights: {recovered_weights}')
                weights_for_plotting = append_zeros(recovered_weights)
                # print(f'p_values: {p_values}')
                # print(f'weights_for_plotting: {weights_for_plotting}')
                plot_input_vectors(weights_for_plotting,
                                # p_values, # my addition
                                figure_directory,
                                title="GLM fit Group " + str(group) + "; Final LL = " +
                                str(loglikelihood_train),
                                save_title='group' + str(group) + '_init' + str(iter),
                                labels_for_plot=labels_for_plot)
                loglikelihood_train_vector.append(loglikelihood_train)
                np.savez(
                    figure_directory + 'variables_of_interest_iter_' + str(iter) +
                    '.npz', loglikelihood_train, recovered_weights)
