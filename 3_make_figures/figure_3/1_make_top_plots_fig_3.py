# Create panels a-c of Figure 3 of Ashwood et al. (2020)
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm # my addition

sys.path.append('../')

from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_was_correct


if __name__ == '__main__':
    # group_subj = "01_00002"
    K = 3

    data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/data_by_subj/'
    # results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/individual_fit/' + group_subj + '/'
    figure_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/figures/figure_3/K_' + str(K) + '/'
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    for group in range(1,4):
        group_str = f'{group:02d}' # which group we are currently looking at 
        

        # subj_list = load_subj_list(data_dir + group_str + '_final_subject_list.npz')
        container = np.load(data_dir + group_str + '_final_subject_list.npz', allow_pickle=True)
        data = [container[key] for key in container]
        subj_list = data[0]

        for subj in tqdm(subj_list, desc=f'Group {group}'):
            group_subj = group_str + '_' + subj

            results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/individual_fit/' + group_subj + '/'

            np.random.seed(59)

            # Fit GLM to data from single animal:
            subj_file = data_dir + group_subj + '_processed.npz'

            # cv_file = results_dir + "/cvbt_folds_model.npz"
            # cvbt_folds_model = load_cv_arr(cv_file)

            # with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
            #     best_init_cvbt_dict = json.load(f)

            # # Get the file name corresponding to the best initialization for given K
            # # value
            # raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
            #                                              results_dir,
            #                                              best_init_cvbt_dict)
            raw_file = results_dir + 'GLM_HMM_K_' + str(K) + '/fold_0/iter_0/glm_hmm_raw_parameters_itr_0' + '.npz'
            hmm_params, lls = load_glmhmm_data(raw_file)

            # Save parameters for initializing individual fits
            weight_vectors = hmm_params[2]
            log_transition_matrix = hmm_params[1][0]
            init_state_dist = hmm_params[0][0]

            # Also get data for subj:
            # inpt, y, session = load_data(data_dir + group_subj + '_processed.npz')
            # container = np.load(data_dir + group_subj + '_processed.npz', allow_pickle=True)
            container = np.load(subj_file, allow_pickle=True)
            data = [container[key] for key in container]
            inpt = data[0]
            y = data[1]
            y = y.astype('int')
            # all_sessions = np.unique(session)
            # Create mask:
            # Identify violations for exclusion:
            violation_idx = np.where(y == -1)[0]
            nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                        inpt.shape[0])
            y[np.where(y == -1), :] = 1
            # inputs, datas, train_masks = partition_data_by_session(
            #     np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask,
            #     session)
            # inputs = inpt
            inputs = np.hstack((inpt, np.ones((len(inpt), 1))))
            datas = y
            train_masks = mask

            # print(f"input: {inputs.shape}, data: {datas.shape}, mask: {train_masks.shape}, params: {hmm_params[2].shape}")

            posterior_probs = get_marginal_posterior([inputs], [datas], [train_masks],
                                                    hmm_params, K, range(K))
            states_max_posterior = np.argmax(posterior_probs, axis=1)

            # sess_to_plot = ["CSHL_008-2019-04-29-001", "CSHL_008-2019-08-07-001",
            #                 "CSHL_008-2019-05-28-001"]
            # sess_to_plot = ["1"]

            cols = ['#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
                    '#999999', '#e41a1c', '#dede00']
            fig = plt.figure(figsize=(6, 5))
            plt.subplots_adjust(wspace=0.2, hspace=0.9)
            # for i, sess in enumerate(sess_to_plot):
            # plt.subplot(3, 3, i + 4)
            plt.subplot(2,1,1)
            # idx_session = np.where(session == sess)
            # this_inpt, this_y = inpt[idx_session[0], :], y[idx_session[0], :]
            this_inpt, this_y = inpt, y
            was_correct, idx_easy = get_was_correct(this_inpt, this_y)
            this_y = this_y[:, 0] + np.random.normal(0, 0.03, len(this_y[:, 0]))
            # plot choice, color by correct/incorrect:
            locs_correct = np.where(was_correct == 1)[0]
            locs_incorrect = np.where(was_correct == 0)[0]
            plt.plot(locs_correct, this_y[locs_correct], 'o', color='black',
                        markersize=2, zorder=3, alpha=0.5)
            plt.plot(locs_incorrect, this_y[locs_incorrect], 'v', color='red',
                        markersize=2, zorder=4, alpha=0.5)

            # states_this_sess = states_max_posterior[idx_session[0]]
            states_this_sess = states_max_posterior
            state_change_locs = np.where(np.abs(np.diff(states_this_sess)) > 0)[0]
            for change_loc in state_change_locs:
                plt.axvline(x=change_loc, color='k', lw=0.5, linestyle='--')
            plt.ylim((-0.13, 1.13))
            # if i == 0:
            plt.xticks([0, len(y)/2, len(y)], ["0", str(len(y)/2), str(len(y))], fontsize=10)
            plt.yticks([0, 1], ["Y", "N"], fontsize=10)
            # else:
            #     plt.xticks([0, 45, 90], ["", "", ""], fontsize=10)
            #     plt.yticks([0, 1], ["", ""], fontsize=10)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            # plt.title("example session " + str(i + 1), fontsize=10)
            plt.title(group_subj, fontsize=10)
            # if i == 0:
            plt.xlabel("trial #", fontsize=10)
            plt.ylabel("choice", fontsize=10)

            # Plot state posterior probabilities
            plt.subplot(2, 1, 2)
            for k in range(K):
                plt.plot(posterior_probs[:, k], label="State " + str(k + 1), lw=1, color=cols[k])

            for change_loc in state_change_locs:
                plt.axvline(x=change_loc, color='k', lw=0.5, linestyle='--')

            plt.xticks([0, len(y)/2, len(y)], ["0", str(len(y)/2), str(len(y))], fontsize=10)
            plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=10)
            plt.ylim((-0.01, 1.01))
            plt.title("Single Session - State Probabilities", fontsize=10)
            plt.xlabel("trial #", fontsize=10)
            plt.ylabel("p(state)", fontsize=10)
            plt.legend()

            # for i, sess in enumerate(sess_to_plot):
            # plt.subplot(3, 3, i + 1)
            # plt.subplot(2,1,2)
            # # idx_session = np.where(session == sess)
            # # this_inpt = inpt[idx_session[0], :]
            # this_inpt = inpt
            # # posterior_probs_this_session = posterior_probs[idx_session[0], :]
            # posterior_probs_this_session = posterior_probs
            # # Plot trial structure for this session too:
            # for k in range(K):
            #     plt.plot(posterior_probs_this_session[:, k],
            #                 label="State " + str(k + 1), lw=1,
            #                 color=cols[k])
            # # states_this_sess = states_max_posterior[idx_session[0]]
            # states_this_sess = states_max_posterior
            # state_change_locs = np.where(np.abs(np.diff(states_this_sess)) > 0)[0]
            # for change_loc in state_change_locs:
            #     plt.axvline(x=change_loc, color='k', lw=0.5, linestyle='--')
            # # if i == 0:
            # plt.xticks([0, 45, 90], ["0", "45", "90"], fontsize=10)
            # plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=10)
            # # else:
            # #     plt.xticks([0, 45, 90], ["", "", ""], fontsize=10)
            # #     plt.yticks([0, 0.5, 1], ["", "", ""], fontsize=10)
            # plt.ylim((-0.01, 1.01))
            # # plt.title("example session " + str(i + 1), fontsize=10)
            # plt.gca().spines['right'].set_visible(False)
            # plt.gca().spines['top'].set_visible(False)
            # # if i == 0:
            # plt.xlabel("trial #", fontsize=10)
            # plt.ylabel("p(state)", fontsize=10)

            # # Now plot avg session:
            # posterior_probs_mat = []
            # for i, sess in enumerate(all_sessions):
            #     idx_session = np.where(session == sess)
            #     posterior_probs_this_session = posterior_probs[idx_session[0], :]
            #     if len(posterior_probs_this_session) == 90:
            #         posterior_probs_mat.append(posterior_probs_this_session)
            # posterior_probs_mat = np.array(posterior_probs_mat)
            # avg_posterior = np.mean(posterior_probs_mat, axis=0)
            # std_dev_posterior = np.std(posterior_probs_mat, axis=0)
            # plt.subplot(3, 3, 7)
            # for k in range(K):
            #     plt.plot(avg_posterior[:, k], label="State " + str(k + 1), lw=1,
            #              color=cols[k])
            #     se = std_dev_posterior[:, k] / np.sqrt(posterior_probs_mat.shape[0])
            #     plt.plot(avg_posterior[:, k] + se, color=cols[k], alpha=0.2)
            #     plt.plot(avg_posterior[:, k] - se, color=cols[k], alpha=0.2)

            # plt.xticks([0, 45, 90], ["", "", ""], fontsize=10)
            # plt.yticks([0, 0.5, 1], ["", "", ""], fontsize=10)
            # plt.ylim((-0.01, 1.01))
            # plt.gca().spines['right'].set_visible(False)
            # plt.gca().spines['top'].set_visible(False)
            # plt.title("avg. session ", fontsize=10)
            # plt.xlabel("trial #", fontsize=10)
            # plt.ylabel("p(state)", fontsize=10)
            # plt.xticks([0, 45, 90], ["0", "45", "90"], fontsize=10)
            # plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=10)
            # fig.savefig(figure_dir + 'fig3abc.pdf')

            fig.savefig(figure_dir + 'fig3_' + group_subj + '.jpg')
            plt.close()
