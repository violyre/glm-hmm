import autograd.numpy as np
import autograd.numpy.random as npr
import os
import matplotlib.pyplot as plt
from glm_utils import load_subj_list, append_zeros

all_labels = ['stim_probe X', 'stim_probe Y', 'stim_probe dist', 'stim_probe angle',
                'stim_1 X', 'stim_1 Y', 'stim_1 dist', 'stim_1 angle',
                'stim_2 X', 'stim_2 Y', 'stim_2 dist', 'stim_2 angle',
                'stim_3 X', 'stim_3 Y', 'stim_3 dist', 'stim_3 angle',
                'prev_resp', 'prev_acc', 'bias']
labels_for_plot = all_labels # temp

if __name__ == '__main__':
    data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/data_by_subj/'
    num_folds = 5
    results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/individual_fit/'
    
    for group in range(1,4):
        group_str = f'{group:02d}' # which group we are currently looking at 
        subj_list = load_subj_list(data_dir + group_str + '_final_subject_list.npz')
        # print(subj_list)

        fig = plt.figure(figsize=(7, 9), dpi=80, facecolor='w', edgecolor='k')
        plt.subplots_adjust(left=0.15,
                            bottom=0.27,
                            right=0.95,
                            top=0.95,
                            wspace=0.3,
                            hspace=0.3)

        for subj in subj_list:
            for fold in range(1):
                this_results_dir = results_dir + group_str + '_' + subj + '/'
                figure_directory = this_results_dir + "GLM/fold_" + str(fold) + '/'
                
                param_file = figure_directory + 'variables_of_interest_iter_' + str(fold) + '.npz'
                container = np.load(param_file)
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
                        plt.plot(range(M + 1), -Ws[j][k], marker='o')
                        plt.plot(range(-1, M + 2), np.repeat(0, M + 3), 'k', alpha=0.2)
                        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
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
                            
                        #plt.ylim((-6,6))

                fig.text(0.04,
                        0.5,
                        "Weight",
                        ha="center",
                        va="center",
                        rotation=90,
                        fontsize=15)
                # fig.suptitle("GLM Weights: " + title, y=0.99, fontsize=14)
                # fig.savefig(figure_directory + 'glm_weights_' + save_title + '.png')

                fig.show() # my addition