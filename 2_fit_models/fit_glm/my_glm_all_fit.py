#  Fit GLM to all IBL data together

import autograd.numpy as np
import autograd.numpy.random as npr
import os
from glm_utils import load_session_fold_lookup, load_data, fit_glm, \
    plot_input_vectors, append_zeros

C = 2  # number of output types/categories
N_initializations = 10 # where does this come from?
npr.seed(65)  # set seed in case of randomization

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

    for group in range(1,4): # iterate through groups 1-3 
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

        for fold in range(num_folds):
            # Subset to relevant covariates for covar set of interest:
            labels_for_plot = ['stim_probe X', 'stim_probe Y', 'stim_probe X*Y', 'stim_1 X', 'stim_1 Y', 'stim_1 X*Y', 'stim_2 X', 'stim_2 Y', 'stim_2 X*Y', 'stim_3 X', 'stim_3 Y', 'stim_3 X*Y', 'prev_resp', 'prev_acc', 'bias']
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

            # M = this_inpt.shape[1]
            M = input.shape[1]
            loglikelihood_train_vector = []

            for iter in range(N_initializations):  # GLM fitting should be
                # independent of initialization, so fitting multiple
                # initializations is a good way to check that everything is
                # working correctly
                loglikelihood_train, recovered_weights = fit_glm([this_input],
                                                                [this_y], M, C)
                print(f'iter {iter}, recovered_weights: {recovered_weights}')
                weights_for_plotting = append_zeros(recovered_weights)
                print(f'weights_for_plotting: {weights_for_plotting}')
                plot_input_vectors(weights_for_plotting,
                                figure_directory,
                                title="GLM fit; Final LL = " +
                                str(loglikelihood_train),
                                save_title='init' + str(iter),
                                labels_for_plot=labels_for_plot)
                loglikelihood_train_vector.append(loglikelihood_train)
                np.savez(
                    figure_directory + 'variables_of_interest_iter_' + str(iter) +
                    '.npz', loglikelihood_train, recovered_weights)
