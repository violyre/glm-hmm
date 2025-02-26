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
    K = 2
    # D, M, C = 1, 3, 2

    for group in range(1,4):
        group_str = f'{group:02d}'

        data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/data_by_subj/'
        overall_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/individual_fit/'
        global_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/global_fit/' 
        figure_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/figures/figure_4/K_' + str(K) + '/'
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        # subj_list = load_subj_list(data_dir + group_str + '_final_subject_list.npz')
        container = np.load(data_dir + group_str + '_final_subject_list.npz', allow_pickle=True)
        data = [container[key] for key in container]
        subj_list = data[0]

        # get_file_name_for_best_model_fold():
        raw_file = global_dir + 'GLM_HMM_K_' + str(K) + '/' + group_str + '_fold_0/iter_0/glm_hmm_raw_parameters_itr_0' + '.npz'
        hmm_params, lls = load_glmhmm_data(raw_file)
        # permutation = calculate_state_permutation(hmm_params)
        # global_weights = -hmm_params[2][permutation]
        global_weights = -hmm_params[2] # ?
        # print(global_weights)

        # fig = plt.figure(figsize=(6, 6))
        fig = plt.figure(figsize=((6+2)*K, 6))
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.95,
                            top=0.95,
                            wspace=0.45,
                            hspace=0.6)
        
        fig.suptitle('Group ' + group_str + ', K=' + str(K), fontsize=14)

        container = np.load(data_dir + group_str + '_final_subject_list.npz', allow_pickle=True)
        data = [container[key] for key in container]
        subj_list = data[0]

        # ==================== WEIGHTS =======================
        cols = [
            '#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00'
        ]

        for k in range(K):
            plt.subplot(1, K, k+1) # plt.subplot(3, 3, k + 4)
            for subj in subj_list:
                results_dir = overall_dir + group_str + '_' + subj + '/'

                # cv_file = results_dir + "/cvbt_folds_model.npz"
                # cvbt_folds_model = load_cv_arr(cv_file)

                # with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
                #     best_init_cvbt_dict = json.load(f)

                # Get the file name corresponding to the best initialization for
                # given K value
                # raw_file = get_file_name_for_best_model_fold(
                #     cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
                raw_file = results_dir + 'GLM_HMM_K_' + str(K) + '/fold_0/iter_0/glm_hmm_raw_parameters_itr_0' + '.npz'
                hmm_params, lls = load_glmhmm_data(raw_file)

                transition_matrix = np.exp(hmm_params[1][0])
                weight_vectors = -hmm_params[2]

                # plot each individual subject's weights
                plt.plot(range(0, len(labels_for_plot)),
                            weight_vectors[k][0][range(0, len(labels_for_plot))],
                            '-o',
                            color=cols[k],
                            lw=1,
                            alpha=0.7,
                            markersize=3,
                            zorder=0)
                
            # plot the global fit
            plt.plot(range(0, len(labels_for_plot)),
                global_weights[k][0][range(0, len(labels_for_plot))],
                '-o',
                color='k',
                lw=1.3,
                alpha=1,
                markersize=3,
                zorder=1,
                label='global fit')

            # if k == 0:
            plt.yticks([-3, 0, 3, 6, 9, 12], fontsize=10)
            plt.xticks(range(0, len(labels_for_plot)), labels_for_plot,
                        fontsize=8,
                        rotation=90)
            plt.ylabel('GLM weight', fontsize=10)
            # else:
            #     plt.yticks([-3, 0, 3, 6, 9, 12], ['', '', '', '', '', ''])
            #     plt.xticks([0, 1, 2, 3], ['', '', '', ''])

            plt.axhline(y=0, color="k", alpha=0.5, ls="--", linewidth=0.75)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.ylim((-4, 14))
            if k == 0:
                plt.legend(fontsize=10,
                            labelspacing=0.2,
                            handlelength=1.4,
                            borderaxespad=0.2,
                            borderpad=0.2,
                            framealpha=0,
                            bbox_to_anchor=(0.2, 0.8))
                
        fig.savefig(figure_dir + 'fig4_' + group_str + '.jpg')
        plt.close()