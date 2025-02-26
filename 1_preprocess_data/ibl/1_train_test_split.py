import numpy as np
from sklearn import preprocessing
import os
import re
import pandas as pd
from collections import defaultdict
from preprocessing_utils import split_train_test, load_data, load_subject_list

if __name__ == '__main__':
    # Set paths
    data_path = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/'
    processed_data_path = os.path.join(data_path, "data_for_cluster/")
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(os.path.join(processed_data_path, "data_by_subj/"), exist_ok=True)

    final_subject_list_all = {}  
    orig_subj_list_len = {}  
    final_subj_list_len = {}  

    for group in range(1, 4):  # Process groups 1, 2, and 3 separately
        group_str = f'{group:02d}'
        pattern = re.compile(f'16_{group_str}_(\\d{{5}})_')  
        subject_start_idx = {}
        subject_end_idx = {}

        master_inpt = []
        master_y = []
        master_trial_fold_lookup_table = []
        subject_tags_all = []

        subject_list = load_subject_list(data_path + 'data_for_cluster/data_by_subj/' + group_str + '_final_subject_list.npz')

        for z, subject in enumerate(subject_list):
            subj_unnormalized_inpt, subj_y = load_data(processed_data_path + f'data_by_subj/{group_str}_{subject}_unnormalized.npz')
            subj_trial_fold_lookup = split_train_test(subj_unnormalized_inpt, split_ratio=0.7)
            np.savez(processed_data_path + 'data_by_subj/' + group_str + '_' + subject + '_trial_fold_lookup.npz', subj_trial_fold_lookup)

            # Get train/test indices
            train_indices = subj_trial_fold_lookup[subj_trial_fold_lookup[:, 1] == 'train', 0].astype(int)
            test_indices = subj_trial_fold_lookup[subj_trial_fold_lookup[:, 1] == 'test', 0].astype(int)

            # Extract features and normalize based on train data stats
            scaler = preprocessing.StandardScaler()
            subj_features = subj_unnormalized_inpt[:, :-1]
            subj_features[train_indices] = scaler.fit_transform(subj_features[train_indices])  # Normalize using training stats
            subj_features[test_indices] = scaler.transform(subj_features[test_indices])  # Apply same transformation

            # Reconstruct full normalized input
            subj_normalized_inpt = np.copy(subj_unnormalized_inpt)
            subj_normalized_inpt[:, :-1] = subj_features

            np.savez(processed_data_path + 'data_by_subj/' + group_str + '_' + subject + '_processed.npz',
                subj_normalized_inpt, subj_y)

            # Append to master dataset
            subject_tags = np.full(len(subj_unnormalized_inpt), subject)
            subject_start_idx[subject] = len(master_inpt)
            master_inpt.append(subj_normalized_inpt)
            master_y.append(subj_y)
            master_trial_fold_lookup_table.append(subj_trial_fold_lookup)
            subject_tags_all.append(subject_tags)
            subject_end_idx[subject] = len(master_inpt) - 1

        # Convert lists to numpy arrays
        master_inpt = np.vstack(master_inpt)
        master_y = np.vstack(master_y)
        master_trial_fold_lookup_table = np.vstack(master_trial_fold_lookup_table)
        subject_tags_all = np.concatenate(subject_tags_all)

        # Save everything as a single file
        np.savez(os.path.join(processed_data_path, f'{group_str}_all_subj_concat.npz'), master_inpt, master_y)
        np.savez(os.path.join(processed_data_path, f'{group_str}_all_subj_concat_trial_fold_lookup.npz'), master_trial_fold_lookup_table)
        np.savez(os.path.join(processed_data_path, f'{group_str}_all_subj_tags.npz'), subject_tags_all)

        # Also save as CSV for readability
        pd.DataFrame(master_inpt).to_csv(os.path.join(processed_data_path, f'{group_str}_all_subj_concat.csv'))
        pd.DataFrame(master_y).to_csv(os.path.join(processed_data_path, f'{group_str}_all_subj_concat_y.csv'))
        pd.DataFrame(master_trial_fold_lookup_table).to_csv(os.path.join(processed_data_path, f'{group_str}_all_subj_concat_trial_fold_lookup.csv'))
        pd.DataFrame(subject_tags_all).to_csv(os.path.join(processed_data_path, f'{group_str}_all_subj_tags.csv'))
        ####
        # write out data from across subjects
        assert np.shape(master_inpt)[0] == np.shape(master_y)[
            0], "inpt and y not same length"
        # assert np.shape(master_correct)[0] == np.shape(master_y)[
        #     0], "correct and y not same length"
        assert len(master_inpt) == \
            np.shape(master_trial_fold_lookup_table)[
                0], "number of total trials and trial fold lookup don't " \
                    "match"
        # assert len(subject_list) == num_subjects, f"{num_subjects} subjects in group 1" # not sure what the point of doing this for us is

        normalized_inpt = np.copy(master_inpt) 

        np.savez(processed_data_path + group_str + '_all_subj_concat.npz',
            normalized_inpt,
            master_y)
        pd.DataFrame(normalized_inpt).to_csv(processed_data_path + group_str + '_all_subj_concat.csv') # also save as csv for readability
        pd.DataFrame(master_y).to_csv(processed_data_path + group_str + '_all_subj_concat_y.csv') # also save master_y as csv
        # np.savez(
        #     processed_data_path + group_str + '_all_subj_concat_unnormalized.npz',
        #     master_inpt, master_y)
        np.savez(
            processed_data_path + group_str + '_all_subj_concat_trial_fold_lookup.npz',
            master_trial_fold_lookup_table)
        pd.DataFrame(master_trial_fold_lookup_table).to_csv(processed_data_path + group_str + '_all_subj_concat_trial_fold_lookup.csv') # save as csv for readability
        # np.savez(processed_data_path + group_str + '_all_subj_concat_correct.npz',
        #     master_correct)
        np.savez(processed_data_path + group_str + '_all_subj_tags.npz', subject_tags_all) # save list of tags for normalized_inpt
        pd.DataFrame(subject_tags_all).to_csv(processed_data_path + group_str + '_all_subj_tags.csv') # save as csv for readability

        # # Now write out normalized data (when normalized across all subjects) for
        # # each subject:        
        # counter = 0
        # for subject in subject_start_idx.keys():
        #     start_idx = subject_start_idx[subject]
        #     end_idx = subject_end_idx[subject]
        #     inpt = normalized_inpt[range(start_idx, end_idx + 1)]
        #     y = master_y[range(start_idx, end_idx + 1)]
        #     # session = master_session[range(start_idx, end_idx + 1)]
        #     counter += inpt.shape[0]
        #     np.savez(processed_data_path + 'data_by_subj/' + group_str + '_' + subject + '_processed.npz',
        #         inpt, y)
        # print(f'counter {counter}, master_inpt.shape[0] {master_inpt.shape[0]}')
        # assert counter == master_inpt.shape[0]

    # print(f'Group {1}: {final_subj_list_len[1]} of {orig_subj_list_len[1]} subjects, {orig_subj_list_len[1]-final_subj_list_len[1]} removed, {final_subj_list_len[1]/orig_subj_list_len[1]*100}% kept')
    # print(f'Group {2}: {final_subj_list_len[2]} of {orig_subj_list_len[2]} subjects, {orig_subj_list_len[2]-final_subj_list_len[2]} removed, {final_subj_list_len[2]/orig_subj_list_len[2]*100}% kept')
    # print(f'Group {3}: {final_subj_list_len[3]} of {orig_subj_list_len[3]} subjects, {orig_subj_list_len[3]-final_subj_list_len[3]} removed, {final_subj_list_len[3]/orig_subj_list_len[3]*100}% kept')

    # print(final_subject_list_all)
    # # save list of all final subjects as dictionary with groups: subj ids
    # json = json.dumps(final_subject_list_all)
    # f = open(processed_data_path + 'data_by_subj/' + 'total_final_subject_list.json', "w")
    # f.write(json)
    # f.close()