#  Fit GLM to all IBL data together
#%%
import autograd.numpy as np
import autograd.numpy.random as npr
import os
from glm_utils import load_session_fold_lookup, load_data, fit_glm, \
    plot_input_vectors, append_zeros, \
    plot_feature_selection_ll, \
    update_features
from tqdm import tqdm # my addition
import statistics # for variance
import matplotlib.pyplot as plt

C = 2  # number of output types/categories
N_initializations = 10 # where does this come from?
npr.seed(65)  # set seed in case of randomization

all_labels = ['stim_probe X', 'stim_probe Y', 'stim_probe dist', 'stim_probe angle',
                'stim_1 X', 'stim_1 Y', 'stim_1 dist', 'stim_1 angle',
                'stim_2 X', 'stim_2 Y', 'stim_2 dist', 'stim_2 angle',
                'stim_3 X', 'stim_3 Y', 'stim_3 dist', 'stim_3 angle',
                'prev_resp', 'prev_acc', 'bias']

doing_feature_selection = True # change this flag if you are using this code to do feature selection or not

# for manual feature selection
features_to_remove = ['stim_probe X', 'stim_probe Y', 'stim_1 X', 'stim_1 Y', 
                      'stim_2 X', 'stim_2 Y', 'stim_3 X', 'stim_3 Y']  # Update this list with features you want to remove

# # Update features and labels based on removal
feat_idxs_to_keep = update_features(features_to_remove, all_labels)
labels_for_plot = [all_labels[i] for i in feat_idxs_to_keep]
print(labels_for_plot)

if 'bias' not in features_to_remove:
    feat_idxs_to_keep = feat_idxs_to_keep[:-1] # remove last term so it doesn't cause an issue with input

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

    if doing_feature_selection:
        ll_vectors_allgroups = [] # store mean ll vectors across all folds for each group
        original_ll_allgroups = [] # store original ll averaged across all folds for each group

    for group in range(1,4): # iterate through groups 1-3 
        group_str = f'{group:02d}' # which group we are currently looking at 

        # Fit GLM to all data
        subj_file = data_dir + group_str + '_all_subj_concat.npz'
        # inpt, y = load_data(subj_file)
        container = np.load(subj_file, allow_pickle=True)
        data = [container[key] for key in container]
        inpt = data[0]
        y = data[1]
        y = y.astype('int')

        # # suggested optimization of above
        # subj_file = os.path.join(data_dir, f'{group_str}_all_subj_concat.npz')
        # container = np.load(subj_file, allow_pickle=True)
        # inpt, y = container['arr_0'], container['arr_1']
        # y = y.astype('int')

        # session_fold_lookup_table = load_session_fold_lookup(
        #     data_dir + 'all_animals_concat_session_fold_lookup.npz')

        # remove features if needed
        inpt = inpt[:, feat_idxs_to_keep]
        print(np.shape(inpt))

        if doing_feature_selection:
            ll_vectors_allfolds = [] # store ll vectors of all folds
            original_loglikelihoods = [] # store original loglikelihoods calculated for each fold

        for fold in range(num_folds):
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
            this_inpt, this_y = inpt[idx_no_viol], y[idx_no_viol] # exclude any violation trials
            # print(f'shape of y: {np.shape(y)} vs shape of this_y: {np.shape(this_y)}')
            # print(f'shape of input: {np.shape(inpt)} vs shape of this_inpt: {np.shape(this_inpt)}')
            
            assert len(np.unique(this_y)) == 2, "choice vector should only include 2 possible values"
            train_size = inpt.shape[0]

            # if not doing feature selection, just plot the regular glm with all features
            if not doing_feature_selection:
                # M = this_inpt.shape[1]
                M = inpt.shape[1]
                loglikelihood_train_vector = []

                for iter in tqdm(range(N_initializations), desc=f'Group {group}, Fold {fold}', unit='init'):  
                    loglikelihood_train, recovered_weights = fit_glm([this_inpt],
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
            elif doing_feature_selection:
                # original_loglikelihoods = [fit_glm([this_inpt], [this_y], inpt.shape[1], C)[0] for _ in range(N_initializations)]
                original_loglikelihood = np.mean([fit_glm([this_inpt], [this_y], inpt.shape[1], C)[0] for _ in range(N_initializations)]) # assume they are all very similar
                print(f'Original loglikelihood for group {group}: {original_loglikelihood}')

                loglikelihood_vectors = []
                if 'bias' in labels_for_plot:
                    all_feats = labels_for_plot[:-1] # remove the bias label so it won't cause an issue in this usage
                else:
                    all_feats = labels_for_plot

                # Iterate over each feature to remove it one at a time
                for feature_to_remove in tqdm(all_feats, desc=f'Group {group}, Fold {fold}, Feature Selection', unit='feature'):
                    feature_indices = update_features([feature_to_remove], all_feats)
                    this_inpt_mod = this_inpt[:, feature_indices]  # Select only the relevant columns
                    print(f"\n Removing feature {feature_to_remove}, keeping indices {feature_indices}")

                    M = len(feature_indices)
                    loglikelihood_train_vector = [
                        fit_glm([this_inpt_mod], [this_y], M, C)[0]
                        for _ in range(N_initializations)
                    ]
                    if statistics.variance(loglikelihood_train_vector)>1:
                        print(f"High variance in vector for variable {feature_to_remove}")
                        input("Press enter to continue")
                    loglikelihood_vectors.append(np.mean(loglikelihood_train_vector))

                    print(f"LL difference: {original_loglikelihood - np.mean(loglikelihood_train_vector)}")
                
                original_loglikelihoods.append(original_loglikelihood) # save the original loglikelihood calculated for this fold

                # save entire loglikelihood_vector for this fold to the variable containing all vectors for all folds
                ll_vectors_allfolds.append(loglikelihood_vectors) # should ultimately store 1 vector for each fold

        if doing_feature_selection:
            assert np.shape(ll_vectors_allfolds)[0] == num_folds, "incorrect number of LL vectors stored"
            assert np.shape(original_loglikelihoods)[0] == num_folds, "incorrect number of original LLs stored"
            original_ll_allgroups.append(np.mean(original_loglikelihoods)) # should ultimately store 1 value for each group

            plot_feature_selection_ll(all_feats,ll_vectors_allfolds,original_ll_allgroups[group-1],num_folds,
                                        directory=results_dir + 'GLM/',
                                        title=f'Group {group}',
                                        save_title=f"feat_select_ll_group_{group}.png",
                                        type='Fold')

            print(f"mean ll vector for group {group}: {np.mean(ll_vectors_allfolds, axis=0)}")
            ll_vectors_allgroups.append(np.mean(ll_vectors_allfolds,axis=0)) # should ultimately store 1 vector for each group

    if not doing_feature_selection: # plot all groups
        fig = plt.figure(figsize=(7, 9), dpi=80, facecolor='w', edgecolor='k')
        plt.subplots_adjust(left=0.15,
                            bottom=0.27,
                            right=0.95,
                            top=0.95,
                            wspace=0.3,
                            hspace=0.3)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")

        for group in range(1,4):
            group_str = f'{group:02d}' # which group we are currently looking at 
            figure_directory = results_dir + 'GLM/' + group_str + '_fold_4' + '/' # take last fold
            glm_vectors_file = figure_directory + 'variables_of_interest_iter_9' + '.npz' # take last iter
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

        print("Non-feature selection run complete")

    elif doing_feature_selection:
        assert np.shape(ll_vectors_allgroups)[0] == 3, "incorrect number of LL vectors stored at end"
        assert np.shape(original_ll_allgroups)[0] == 3, "incorrect number of original LLs stored"

        np.savez(results_dir + 'GLM/' + 'feat_select_all_lls.npz', ll_vectors_allgroups, original_ll_allgroups)

        plot_feature_selection_ll(all_feats,ll_vectors_allgroups,original_ll_allgroups,num_folds,
                                    directory=results_dir + 'GLM/',
                                    title=f'All Groups, Averaged Across Folds',
                                    save_title="feat_select_ll_allgroups_diff.png",
                                    type='GroupDiff')
        
        print("Feature selection complete")
