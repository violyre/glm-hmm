import autograd.numpy as np
import matplotlib.pyplot as plt
from glm_utils import append_zeros, update_features
import json

doing_feature_selection = False # change this flag if you are using this code to do feature selection or not

if __name__ == '__main__':
    results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/global_fit/'

    with open('C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/labels_for_plot.json', 'r') as f:
        labels_for_plot = json.load(f)
    print(labels_for_plot)

    fig = plt.figure(figsize=(7, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.15,
                        bottom=0.27,
                        right=0.95,
                        top=0.95,
                        wspace=0.3,
                        hspace=0.3)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    # plt.ylim((-4,4))
    plt.ylim((-3,8))

    for group in range(1,4):
        group_str = f'{group:02d}' # which group we are currently looking at 
        figure_directory = results_dir + 'GLM/' + group_str + '_fold_0' + '/' # take first fold
        glm_vectors_file = figure_directory + 'variables_of_interest_iter_0' + '.npz' # take first iter
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
                            ['Stimulus', 'Past Choice', 'Bias'], # ashwood's original labels
                            rotation='90',
                            fontsize=12)
                    
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

    plt.show()