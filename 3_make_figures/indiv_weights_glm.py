import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    load_correct_incorrect_mat, load_animal_list, permute_transition_matrix,\
    calculate_state_permutation, get_file_name_for_best_model_fold, \
    partition_data_by_session, create_violation_mask, \
    get_marginal_posterior, get_global_weights, get_global_trans_mat

data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/'
with open(data_dir + 'labels_for_plot.json', 'r') as f:
    labels_for_plot = json.load(f)
print(labels_for_plot)

if __name__ == '__main__':
    for group in range(1,4):
        group_str = f'{group:02d}'

        data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/data_by_subj/'
        global_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/global_fit/GLM/' 
        figure_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/figures/figure_4/K_1' + '/'
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        # subj_list = load_subj_list(data_dir + group_str + '_final_subject_list.npz')
        container = np.load(data_dir + group_str + '_final_subject_list.npz', allow_pickle=True)
        data = [container[key] for key in container]
        subj_list = data[0]

        # get_file_name_for_best_model_fold():
        raw_file = global_dir + group_str + '_fold_0/variables_of_interest_iter_0.npz'
        container = np.load(raw_file)
        data = [container[key] for key in container]
        loglikelihood_train = data[0]
        recovered_weights = data[1]
        print(recovered_weights)
        # Ws = append_zeros(recovered_weights)

        # hmm_params, lls = load_glmhmm_data(raw_file)
        # print(hmm_params)
        # permutation = calculate_state_permutation(hmm_params)
        # global_weights = -hmm_params[2][permutation]
        # global_weights = -hmm_params[2] # ?
        # print(global_weights)

        # fig = plt.figure(figsize=(6, 6))
        fig = plt.figure(figsize=(6, 6))
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.95,
                            top=0.95,
                            wspace=0.45,
                            hspace=0.6)
        
        fig.suptitle('Group ' + group_str + ', GLM', fontsize=14)

        container = np.load(data_dir + group_str + '_final_subject_list.npz', allow_pickle=True)
        data = [container[key] for key in container]
        subj_list = data[0]

        # ==================== WEIGHTS =======================
        cols = [
            '#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00'
        ]

        for subj in subj_list:
            transition_matrix = np.exp(hmm_params[1][0])
            weight_vectors = -hmm_params[2]

            # plot each individual subject's weights
            plt.plot(range(0, len(labels_for_plot)),
                        weight_vectors[0][0][range(0, len(labels_for_plot))],
                        '-o',
                        color=cols[0],
                        lw=1,
                        alpha=0.7,
                        markersize=3,
                        zorder=0)
                
        # plot the global fit
        plt.plot(range(0, len(labels_for_plot)),
            global_weights[0][0][range(0, len(labels_for_plot))],
            '-o',
            color='k',
            lw=1.3,
            alpha=1,
            markersize=3,
            zorder=1,
            label='global fit')

        plt.yticks([-3, 0, 3, 6, 9, 12], fontsize=10)
        plt.xticks(range(0, len(labels_for_plot)), labels_for_plot,
                    fontsize=8,
                    rotation=90)
        plt.ylabel('GLM weight', fontsize=10)

        plt.axhline(y=0, color="k", alpha=0.5, ls="--", linewidth=0.75)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.ylim((-4, 14))
        plt.legend(fontsize=10,
                    labelspacing=0.2,
                    handlelength=1.4,
                    borderaxespad=0.2,
                    borderpad=0.2,
                    framealpha=0,
                    bbox_to_anchor=(0.2, 0.8))
                
        fig.savefig(figure_dir + 'fig4_' + group_str + '.jpg')
        plt.close()