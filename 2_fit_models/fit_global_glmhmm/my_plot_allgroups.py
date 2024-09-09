import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glm_hmm_utils import update_features

all_labels = ['stim_probe X', 'stim_probe Y', 'stim_probe dist', 'stim_probe angle',
                'stim_1 X', 'stim_1 Y', 'stim_1 dist', 'stim_1 angle',
                'stim_2 X', 'stim_2 Y', 'stim_2 dist', 'stim_2 angle',
                'stim_3 X', 'stim_3 Y', 'stim_3 dist', 'stim_3 angle',
                'prev_resp', 'prev_acc', 'bias']

doing_feature_selection = True # change this flag if you are using this code to do feature selection or not

# for manual feature selection
features_to_remove = ['stim_probe X', 'stim_probe Y', 'stim_1 X', 'stim_1 Y', 
                      'stim_2 X', 'stim_2 Y', 'stim_3 X', 'stim_3 Y']  # Update this list with features you want to remove

# Update features and labels based on removal
feat_idxs_to_keep = update_features(features_to_remove, all_labels)
labels_for_plot = [all_labels[i] for i in feat_idxs_to_keep]
print(labels_for_plot)
if 'bias' not in features_to_remove:
    feat_idxs_to_keep = feat_idxs_to_keep[:-1] # remove last term so it doesn't cause an issue with input

if __name__ == '__main__':
    data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/'
    results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/global_fit/'
    save_directory = data_dir + "best_global_params/"

    # labels_for_plot = ['stim_probe X', 'stim_probe Y', 'stim_probe dist', 'stim_probe angle',
    #             'stim_1 X', 'stim_1 Y', 'stim_1 dist', 'stim_1 angle',
    #             'stim_2 X', 'stim_2 Y', 'stim_2 dist', 'stim_2 angle',
    #             'stim_3 X', 'stim_3 Y', 'stim_3 dist', 'stim_3 angle',
    #             'prev_resp', 'prev_acc', 'bias']
    
    all_weights = []
    
    for group in range(1,4): # iterate through groups 1-3 
        group_str = f'{group:02d}'
        K = 2
        best_params = save_directory + 'best_params_' + group_str + '_K_' + str(K) + '.npz'

        container = np.load(best_params, allow_pickle=True)
        data = [container[key] for key in container]
        params_for_individual_initialization = data[0]
        weight_vectors = params_for_individual_initialization[2]
        if group is 1: # manually flip states for groups that differ in state labeling
            temp = np.copy(weight_vectors[1])
            weight_vectors[1] = np.copy(weight_vectors[0])
            weight_vectors[0] = np.copy(temp)
        all_weights.append(weight_vectors)
        print(f"weights for group {group}: {weight_vectors}")

    print(f"all weights: {all_weights}")
    print(f"group 1: {all_weights[1-1][1][0]}")
    
    # Plot these too:
    cols = ["#e74c3c", "#15b01a", "#7e1e9c", "#3498db", "#f97306"]
    linestyles = ["solid", "dashed", "dotted"]
    M = weight_vectors.shape[2] - 1

    for k in range(K):
        fig = plt.figure(figsize=(4 * 4, 8),
                dpi=80,
                facecolor='w',
                edgecolor='k')
        for g in range(1,4):
            plt.plot(range(M + 1),
                    -all_weights[g-1][k][0],
                    marker='o',
                    label='Group ' + str(g) + ', State ' + str(k + 1),
                    color=cols[g+1],
                    lw=4,
                    linestyle=linestyles[k])
        plt.xticks(list(range(0, len(labels_for_plot))),
                labels_for_plot,
                rotation='20',
                fontsize=24,
                ha='right')
        plt.yticks(fontsize=30)
        plt.legend(fontsize=15)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        # plt.ylim((-3, 14))
        plt.ylabel("Weight", fontsize=30)
        plt.xlabel("Covariate", fontsize=30, labelpad=20)
        plt.title("GLM Weights, State " + str(k+1), fontsize=40)
        plt.show()
            
        fig.savefig(results_dir + 'allgroups' + '_K_' +
                            str(K) + '_state_' + str(k+1) + '_iter_' + '0' + '.png')