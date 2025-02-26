#  Fit GLM to all IBL data together
#%%
import autograd.numpy as np
import autograd.numpy.random as npr
import os
import pandas as pd
from glm_utils import load_session_fold_lookup, load_data, load_tags, fit_glm, \
    plot_feature_selection_ll, \
    update_features
from tqdm import tqdm # my addition
import statistics # for variance
import matplotlib.pyplot as plt
import json

C = 2  # number of output types/categories
N_initializations = 10 # where does this come from?
npr.seed(65)  # set seed in case of randomization

train_test_split = False # change this flag if you want to split train/test here

if __name__ == '__main__':
    data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/'
    num_folds = 5 

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

    ll_vectors_allgroups = [] # store mean ll vectors across all folds for each group
    original_ll_allgroups = [] # store original ll averaged across all folds for each group

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

        # remove features if needed
        # inpt = inpt[:, feat_idxs_to_keep]
        # print(np.shape(inpt))

        ll_vectors_allfolds = [] # store ll vectors of all folds
        original_loglikelihoods = [] # store original loglikelihoods calculated for each fold

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
            train_size = inpt.shape[0]

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
    
    assert np.shape(ll_vectors_allgroups)[0] == 3, "incorrect number of LL vectors stored at end"
    assert np.shape(original_ll_allgroups)[0] == 3, "incorrect number of original LLs stored"

    np.savez(results_dir + 'GLM/' + 'feat_select_all_lls.npz', ll_vectors_allgroups, original_ll_allgroups)

    plot_feature_selection_ll(all_feats,ll_vectors_allgroups,original_ll_allgroups,num_folds,
                                directory=results_dir + 'GLM/',
                                title=f'All Groups, Averaged Across Folds',
                                save_title="feat_select_ll_allgroups_diff.png",
                                type='GroupDiff')
    
    print("Feature selection complete")
