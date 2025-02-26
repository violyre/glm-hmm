# Create a matrix of size num_models x num_folds containing
# normalized loglikelihood for both train and test splits
import json

import numpy as np
from post_processing_utils import load_data, load_session_fold_lookup, \
    prepare_data_for_cv, calculate_baseline_test_ll, \
    calculate_glm_test_loglikelihood, calculate_cv_bit_trial, \
    return_glmhmm_nll, return_lapse_nll
from post_processing_utils import create_violation_mask, \
    update_features # my addition

doing_feature_selection = False # change this flag if you are using this code to do feature selection or not
train_test_split = False # change this flag if you want to split train/test here

if __name__ == '__main__':
    data_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/data_for_cluster/'
    results_dir = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/results/global_fit/'

    with open(data_dir + 'labels_for_plot.json', 'r') as f:
        labels_for_plot = json.load(f)
    print(labels_for_plot)

    for group in range(1,4):
        group_str = f'{group:02d}'
        print(f"For group {group}:")

        # Load data
        inpt, y = load_data(data_dir + 'all_subj_concat.npz')
        # container = np.load(data_dir + group_str + '_all_subj_concat.npz', allow_pickle=True)
        # data = [container[key] for key in container]
        # inpt = data[0]
        # y = data[1]
        # y = y.astype('int')

        # # remove features if needed
        # inpt = inpt[:, feat_idxs_to_keep]
        # print(np.shape(inpt))

        # Parameters
        C = 2  # number of output classes
        num_folds = 5  # number of folds
        D = 1  # number of output dimensions
        K_max = 5  # maximum number of latent states
        num_models = K_max + 2  # model for each latent + 2 lapse models

        subj_preferred_model_dict = {}
        models = ["GLM", "Lapse_Model", "GLM_HMM"]

        cvbt_folds_model = np.zeros((num_models, num_folds))
        cvbt_train_folds_model = np.zeros((num_models, num_folds))

        # Save best initialization for each model-fold combination
        best_init_cvbt_dict = {}

        for fold in range(1): #range(num_folds):
            trial_fold_lookup_table = load_session_fold_lookup(data_dir + group_str + '_all_subj_concat_trial_fold_lookup.npz')

            # # prepare_data_for_cv(): 
            # violation_idx = np.where(y == -1)[0]
            # nonviolation_idx, nonviolation_mask = create_violation_mask(violation_idx, inpt.shape[0])

            # # get_train_test_dta():
            # test_trials = np.where(trial_fold_lookup_table[:, 1] == "test")[0]
            # train_trials = np.where(trial_fold_lookup_table[:, 1] == "train")[0]
            # idx_test = [trial for trial in test_trials] 
            # idx_train = [trial for trial in train_trials] 
            # test_inpt, test_y, test_nonviolation_mask = inpt[idx_test, :], y[idx_test, :], nonviolation_mask[idx_test]
            # train_inpt, train_y, train_nonviolation_mask = inpt[idx_train, :], y[idx_train,:], nonviolation_mask[idx_train]

            # M = train_inpt.shape[1]
            # n_test = np.sum(test_nonviolation_mask == 1)
            # n_train = np.sum(train_nonviolation_mask == 1)

            test_inpt, test_y, test_nonviolation_mask, \
            train_inpt, train_y, train_nonviolation_mask, M,\
            n_test, n_train = prepare_data_for_cv(
                inpt, y, trial_fold_lookup_table)

            ll0 = calculate_baseline_test_ll(
                train_y[train_nonviolation_mask == 1, :],
                test_y[test_nonviolation_mask == 1, :], C)
            ll0_train = calculate_baseline_test_ll(
                train_y[train_nonviolation_mask == 1, :],
                train_y[train_nonviolation_mask == 1, :], C)
            for model in models:
                print("model = " + str(model))
                if model == "GLM":
                    # Load parameters and instantiate a new GLM object with
                    # these parameters
                    glm_weights_file = results_dir + '/GLM/' + group_str + '_fold_' + str(
                        fold) + '/variables_of_interest_iter_0.npz'
                    ll_glm = calculate_glm_test_loglikelihood(
                        glm_weights_file, test_y[test_nonviolation_mask == 1, :],
                        test_inpt[test_nonviolation_mask == 1, :], M, C)
                    ll_glm_train = calculate_glm_test_loglikelihood(
                        glm_weights_file, train_y[train_nonviolation_mask == 1, :],
                        train_inpt[train_nonviolation_mask == 1, :], M, C)
                    cvbt_folds_model[0, fold] = calculate_cv_bit_trial(
                        ll_glm, ll0, n_test)
                    cvbt_train_folds_model[0, fold] = calculate_cv_bit_trial(
                        ll_glm_train, ll0_train, n_train)
                elif model == "Lapse_Model":
                    # One lapse parameter model:
                    cvbt_folds_model[1, fold], cvbt_train_folds_model[
                        1,
                        fold], _, _ = return_lapse_nll(inpt, y,
                                                    trial_fold_lookup_table,
                                                    fold, 1, results_dir, C)
                    # Two lapse parameter model:
                    cvbt_folds_model[2, fold], cvbt_train_folds_model[
                        2,
                        fold], _, _ = return_lapse_nll(inpt, y, 
                                                    trial_fold_lookup_table,
                                                    fold, 2, results_dir, C)
                elif model == "GLM_HMM":
                    for K in range(2, K_max + 1):
                        print("K = " + str(K))
                        model_idx = 3 + (K - 2)
                        cvbt_folds_model[model_idx, fold], \
                        cvbt_train_folds_model[
                            model_idx, fold], _, _, init_ordering_by_train = \
                            return_glmhmm_nll(
                                np.hstack((inpt, np.ones((len(inpt), 1)))), y,
                                trial_fold_lookup_table, fold,
                                K, D, C, results_dir)
                        # Save best initialization to dictionary for later:
                        key_for_dict = '/GLM_HMM_K_' + str(K) + '/fold_' + str(
                            fold)
                        best_init_cvbt_dict[key_for_dict] = int(
                            init_ordering_by_train[0])
        # Save best initialization directories across animals, folds and models
        # (only GLM-HMM):
        print(cvbt_folds_model)
        print(cvbt_train_folds_model)
        json_dump = json.dumps(best_init_cvbt_dict)
        f = open(results_dir + "/best_init_cvbt_dict.json", "w")
        f.write(json_dump)
        f.close()
        # Save cvbt_folds_model as numpy array for easy parsing across all
        # models and folds
        np.savez(results_dir + "/cvbt_folds_model.npz", cvbt_folds_model)
        np.savez(results_dir + "/cvbt_train_folds_model.npz",
                cvbt_train_folds_model)
