#  Fit GLM to all IBL data together
#%%
import autograd.numpy as np
import autograd.numpy.random as npr
import os
import pandas as pd
from glm_utils import load_session_fold_lookup, load_data, load_tags, fit_glm, \
    plot_input_vectors, append_zeros
from tqdm import tqdm # my addition
import statistics # for variance
import matplotlib.pyplot as plt
import json

C = 2  # number of output types/categories
N_initializations = 10 # where does this come from?
npr.seed(65)  # set seed in case of randomization

train_test_split = True # change this flag if you want to split train/test here

if __name__ == '__main__':
    data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/'
    num_folds = 1 #5 

    # # for use with glm fit for subjects separately 
    # container = np.load(data_dir + 'data_by_subj/subject_list.npz', allow_pickle=True)
    # data = [container[key] for key in container]
    # subject_list = data[0]

    with open(data_dir + 'labels_for_plot.json', 'r') as f:
        labels_for_plot = json.load(f)
    print(labels_for_plot)

    # Create directory for results:
    results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/global_fit/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for group in range(1,4): # iterate through groups 1-3 
        group_str = f'{group:02d}' # which group we are currently looking at 

        # Fit GLM to all data
        subj_file = data_dir + group_str + '_all_subj_concat.npz'
        tags_file = data_dir + group_str + '_all_subj_tags.npz'
        inpt, y = load_data(subj_file)
        y = y.astype('int')
        tags = load_tags(tags_file)
        
        trial_fold_lookup_table = load_session_fold_lookup(data_dir + group_str + '_all_subj_concat_trial_fold_lookup.npz')

        # # suggested optimization of above
        # subj_file = os.path.join(data_dir, f'{group_str}_all_subj_concat.npz')
        # container = np.load(subj_file, allow_pickle=True)
        # inpt, y = container['arr_0'], container['arr_1']
        # y = y.astype('int')

        for fold in range(num_folds):
            figure_directory = results_dir + 'GLM/' + group_str + '_fold_' + str(fold) + '/'
            print(figure_directory)
            if not os.path.exists(figure_directory):
                os.makedirs(figure_directory)

            idx_no_viol = np.where(y[:,0] != -1) # exclude any violation trials
            if train_test_split: 
                trials_to_keep = np.where(trial_fold_lookup_table[:,1] == "train")[0] # indices in lookup table that correspond to "train"
                idx_this_fold = [id for id in trials_to_keep if y[id,0] != -1]

                this_inpt, this_y = inpt[idx_this_fold], y[idx_this_fold] 
                this_tags = tags[idx_this_fold]
            else:
                this_inpt, this_y = inpt[idx_no_viol], y[idx_no_viol] 
                this_tags = tags[idx_no_viol]
            assert len(np.unique(this_y)) == 2, "choice vector should only include 2 possible values"
            # train_size = inpt.shape[0]

            # if not doing feature selection, just plot the regular glm with all features
            M = this_inpt.shape[1]
            loglikelihood_train_vector = []

            for iter in tqdm(range(N_initializations), desc=f'Group {group}, Fold {fold}', unit='init'):  
                loglikelihood_train, recovered_weights = fit_glm([this_inpt],
                                                                [this_y],
                                                                [this_tags],
                                                                M, C)
                weights_for_plotting = append_zeros(recovered_weights)
                # print(f'p_values: {p_values}')
                plot_input_vectors(weights_for_plotting,
                                # p_values, # my addition
                                figure_directory,
                                title="GLM fit Group " + str(group) + "; Final LL = " +
                                str(loglikelihood_train),
                                save_title='group' + str(group) + '_init' + str(iter),
                                labels_for_plot=labels_for_plot)
                loglikelihood_train_vector.append(loglikelihood_train)
                np.savez(figure_directory + 'variables_of_interest_iter_' + str(iter) + '.npz', loglikelihood_train, recovered_weights)
                # print(f"saved {figure_directory + 'variables_of_interest_iter_' + str(iter) + '.npz'}")
            plt.close('all') # close all figures after each group is done to save memory

            # # suggested optimization of above
            # loglikelihood_train_vector = [
            #     fit_glm([this_inpt], [this_y], M, C)[0]
            #     for _ in tqdm(range(N_initializations), desc=f'Group {group_str}, Fold 1', unit='init')
            # ]
            # weights_for_plotting = append_zeros(fit_glm([this_inpt], [this_y], M, C)[1])
            # plot_input_vectors(weights_for_plotting, figure_directory, title=f"GLM fit Group {group}; Final LL = {np.mean(loglikelihood_train_vector)}",
            #                 save_title=f'group{group}_init', labels_for_plot=labels_for_plot)

            # np.savez(os.path.join(figure_directory, f'variables_of_interest_iter_{N_initializations}.npz'),
            #         loglikelihood_train_vector, weights_for_plotting)

    # plot all groups
    fig = plt.figure(figsize=(7, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.15,
                        bottom=0.27,
                        right=0.95,
                        top=0.95,
                        wspace=0.3,
                        hspace=0.3)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")

    for fold in [0,1]: #range(num_folds):
        for group in range(1,4):
            group_str = f'{group:02d}' # which group we are currently looking at 
            figure_directory = results_dir + 'GLM/' + group_str + '_fold_' + str(fold) + '/' # take first fold
            glm_vectors_file = figure_directory + 'variables_of_interest_iter_' + str(fold) + '.npz' # take first iter
            container = np.load(glm_vectors_file)
            data = [container[key] for key in container]
            loglikelihood_train = data[0]
            recovered_weights = data[1]
            Ws = append_zeros(recovered_weights)

            K = Ws.shape[0]
            K_prime = Ws.shape[1]
            M = Ws.shape[2] - 1

            for j in range(K):
                for k in range(K_prime - 1):
                    # plt.subplot(K, K_prime, 1+j*K_prime+k)
                    plt.plot(range(M + 1), -Ws[j][k], marker='o', label=f'Group {group}')
                    plt.plot(range(-1, M + 2), np.repeat(0, M + 3), 'k', alpha=0.2)
                    if len(labels_for_plot) > 0:
                        plt.xticks(list(range(0, len(labels_for_plot))),
                                labels_for_plot,
                                rotation='90',
                                fontsize=12)
                    else:
                        plt.xticks(list(range(0, 3)),
                                ['Stimulus', 'Past Choice', 'Bias'],
                                rotation='90',
                                fontsize=12)
                        
        plt.ylim((-6,6))
        plt.legend()
        
        fig.text(0.04,
                0.5,
                "Weight",
                ha="center",
                va="center",
                rotation=90,
                fontsize=15)
        fig.suptitle("GLM Weights, All Groups", y=0.99, fontsize=14)
        fig.savefig(results_dir + 'GLM/' + 'glm_weights_allgroups' + '.png')
        plt.close()

    print("Non-feature selection run complete")
