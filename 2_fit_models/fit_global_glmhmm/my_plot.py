# Save best parameters from IBL global fits (for K = 2 to 5) to initialize
# each animal's model
import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from post_processing_utils import load_glmhmm_data, load_cv_arr, \
    create_cv_frame_for_plotting, get_file_name_for_best_model_fold, \
    permute_transition_matrix, calculate_state_permutation


if __name__ == '__main__':
    data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/'
    results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/global_fit/'
    save_directory = data_dir + "best_global_params/"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # labels_for_plot = ['stim', 'pc', 'wsls', 'bias']
    labels_for_plot = ['stim_probe X', 'stim_probe Y', 'stim_probe X*Y', 'stim_1 X', 'stim_1 Y', 'stim_1 X*Y', 'stim_2 X', 'stim_2 Y', 'stim_2 X*Y', 'stim_3 X', 'stim_3 Y', 'stim_3 X*Y', 'prev_resp', 'prev_acc', 'bias']

    # cv_file = results_dir + "/cvbt_folds_model.npz"
    # cvbt_folds_model = load_cv_arr(cv_file)
    
    raw_file_K2 = results_dir + 'GLM_HMM_K_2/fold_0/iter_0/glm_hmm_raw_parameters_itr_0.npz'
    raw_file_K3 = results_dir + 'GLM_HMM_K_3/fold_0/iter_0/glm_hmm_raw_parameters_itr_0.npz'
    # raw_file_K4 = results_dir + 'GLM_HMM_K_4/fold_0/iter_0/glm_hmm_raw_parameters_itr_0.npz'
    # raw_file_K5 = results_dir + 'GLM_HMM_K_5/fold_0/iter_0/glm_hmm_raw_parameters_itr_0.npz'
    # files = [raw_file_K2, raw_file_K3, raw_file_K4, raw_file_K5]
    files = [raw_file_K2, raw_file_K3]

    for K in range(2,4): #range(2,6):
        raw_file = files[K-2]
        # raw_file = raw_file_K2
        container = np.load(raw_file, allow_pickle=True)
        data = [container[key] for key in container]
        hmm_params = data[0]
        lls = data[1]

        # Calculate permutation
        # permutation = calculate_state_permutation(hmm_params)
        # print(permutation)

        # copied from permutation for now (for K=2)
        # GLM weights (note: we have to take negative, because we are interested
        # in weights corresponding to p(y = 1) = 1/(1+e^(-w.x)), but returned
        # weights from
        # code are w such that p(y = 1) = e(w.x)/1+e(w.x))
        glm_weights = -hmm_params[2]
        permutation = np.argsort(-glm_weights[:, 0, 0])

        # Save parameters for initializing individual fits
        weight_vectors = hmm_params[2][permutation]
        log_transition_matrix = permute_transition_matrix(
            hmm_params[1][0], permutation)
        init_state_dist = hmm_params[0][0][permutation]
        params_for_individual_initialization = [[init_state_dist],
                                                [log_transition_matrix],
                                                weight_vectors]

        np.savez(
            save_directory + 'best_params_K_' + str(K) + '.npz',
            params_for_individual_initialization)

        # Plot these too:
        cols = ["#e74c3c", "#15b01a", "#7e1e9c", "#3498db", "#f97306"]
        fig = plt.figure(figsize=(4 * 8, 10),
                         dpi=80,
                         facecolor='w',
                         edgecolor='k')
        plt.subplots_adjust(left=0.1,
                            bottom=0.24,
                            right=0.95,
                            top=0.7,
                            wspace=0.8,
                            hspace=0.5)
        plt.subplot(1, 2, 1)
        M = weight_vectors.shape[2] - 1
        for k in range(K):
            plt.plot(range(M + 1),
                     -weight_vectors[k][0],
                     marker='o',
                     label='State ' + str(k + 1),
                     color=cols[k],
                     lw=4)
        plt.xticks(list(range(0, len(labels_for_plot))),
                   labels_for_plot,
                   rotation='20',
                   fontsize=24)
        plt.yticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        # plt.ylim((-3, 14))
        plt.ylabel("Weight", fontsize=30)
        plt.xlabel("Covariate", fontsize=30, labelpad=20)
        plt.title("GLM Weights: Choice = R", fontsize=40)

        plt.subplot(1, 2, 2)
        transition_matrix = np.exp(log_transition_matrix)
        plt.imshow(transition_matrix, vmin=0, vmax=1)
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[1]):
                text = plt.text(j,
                                i,
                                np.around(transition_matrix[i, j],
                                          decimals=3),
                                ha="center",
                                va="center",
                                color="k",
                                fontsize=30)
        plt.ylabel("Previous State", fontsize=30)
        plt.xlabel("Next State", fontsize=30)
        plt.xlim(-0.5, K - 0.5)
        plt.ylim(-0.5, K - 0.5)
        plt.xticks(range(0, K), ('1', '2', '3', '4', '4', '5', '6', '7',
                                 '8', '9', '10')[:K],
                   fontsize=30)
        plt.yticks(range(0, K), ('1', '2', '3', '4', '4', '5', '6', '7',
                                 '8', '9', '10')[:K],
                   fontsize=30)
        plt.title("Retrieved", fontsize=40)

        # plt.subplot(1, 3, 3)
        # cols = [
        #     "#7e1e9c", "#0343df", "#15b01a", "#bf77f6", "#95d0fc",
        #     "#96f97b"
        # ]
        # cv_file = results_dir + "/cvbt_folds_model.npz"
        # data_for_plotting_df, loc_best, best_val, glm_lapse_model = \
        #     create_cv_frame_for_plotting(
        #     cv_file)
        # cv_file_train = results_dir + "/cvbt_train_folds_model.npz"
        # train_data_for_plotting_df, train_loc_best, train_best_val, \
        # train_glm_lapse_model = create_cv_frame_for_plotting(
        #     cv_file_train)

        # glm_lapse_model_cvbt_means = np.mean(glm_lapse_model, axis=1)
        # train_glm_lapse_model_cvbt_means = np.mean(train_glm_lapse_model,
        #                                            axis=1)
        # g = sns.lineplot(
        #     data_for_plotting_df['model'],
        #     data_for_plotting_df['cv_bit_trial'],
        #     err_style="bars",
        #     mew=0,
        #     color=cols[0],
        #     marker='o',
        #     ci=68,
        #     label="test",
        #     alpha=1,
        #     lw=4)
        # sns.lineplot(
        #     train_data_for_plotting_df['model'],
        #     train_data_for_plotting_df['cv_bit_trial'],
        #     err_style="bars",
        #     mew=0,
        #     color=cols[1],
        #     marker='o',
        #     ci=68,
        #     label="train",
        #     alpha=1,
        #     lw=4)
        # plt.xlabel("Model", fontsize=30)
        # plt.ylabel("Normalized LL", fontsize=30)
        # plt.xticks([0, 1, 2, 3, 4],
        #            ['1 State', '2 State', '3 State', '4 State', '5 State'],
        #            rotation=45,
        #            fontsize=24)
        # plt.yticks(fontsize=15)
        # plt.axhline(y=glm_lapse_model_cvbt_means[2],
        #             color=cols[2],
        #             label="Lapse (test)",
        #             alpha=0.9,
        #             lw=4)
        # plt.legend(loc='upper right', fontsize=30)
        # plt.tick_params(axis='y')
        # plt.yticks([0.2, 0.3, 0.4, 0.5], fontsize=30)
        # plt.ylim((0.2, 0.55))
        # plt.title("Model Comparison", fontsize=40)
        # fig.tight_layout()

        fig.savefig(results_dir + 'K_' +
                    str(K) + '_iter_' + '0' + '.png')