# Continue preprocessing of IBL dataset and create design matrix for GLM-HMM
import numpy as np
from sklearn import preprocessing
import numpy.random as npr
import os
import json
from collections import defaultdict
# from preprocessing_utils import load_animal_list, load_animal_eid_dict, \
#     get_all_unnormalized_data_this_session, create_train_test_sessions
import re # regex
import pandas as pd 
from preprocessing_utils import create_previous_choice_vector
import math
from preprocessing_utils import split_train_test # my function

# npr.seed(65)
delete_first_trial = True # change this flag if you don't want to delete the first trial

if __name__ == '__main__':
    data_path = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/'
    # create directory for saving data:
    if not os.path.exists(data_path + "partially_processed/"):
        os.makedirs(data_path + "partially_processed/")

    os.chdir(data_path) # search in correct directory
    final_subject_list_all = {} # empty dictionary to store all the subj from each group that we keep
    orig_subj_list_len = {} # store length of original subject lists for each group to compare to final subject count
    final_subj_list_len = {} #store length of final subject lists for each group to compare to original subject count 

    for group in range(1,4): # go through groups 1, 2, and 3 and save them separately
        group_str = f'{group:02d}'

        subject_list = [] # store subject IDs
        # pattern = re.compile(r'16_(\d{2})_(\d{5})_(\d{2})') # regex pattern

        pattern = re.compile(f'16_{group_str}_(\\d{{5}})_') # regex pattern for group 1
        subject_ids_dict = defaultdict(list) #why not {}? # create dictionary to store dataframes for each subject ID

        # get list of subject IDs for group 1
        for filename in os.listdir(data_path):
            if filename.endswith("_trialdata.csv"):
                match = pattern.search(filename)
                if match:
                    # chronicity = match.group(2) # chronicity -- ignoring this for now?
                    subject = match.group(1) # ID number 

                    if subject not in subject_list:
                        subject_list.append(subject)

                    subject_ids_dict[subject] = filename # store filename for subject
        np.savez('partially_processed/subject_list_' + group_str + '.npz', subject_list)
        orig_subj_list_len[group] = len(subject_list)

        # Create directories for saving data:
        processed_data_path = data_path + "data_for_cluster/"
        if not os.path.exists(processed_data_path):
            os.makedirs(processed_data_path)
        # Also create a subdirectory for storing each individual subject's data:
        if not os.path.exists(processed_data_path + "data_by_subj/"):
            os.makedirs(processed_data_path + "data_by_subj/")

        wm_only = True # change to False if you want to store all trial types, including CTL

        # Identify idx in master array where each subject's data starts and ends:
        subject_start_idx = {}
        subject_end_idx = {}

        final_subject_ids_dict = defaultdict(list) # should I do {}?

        if not os.path.exists(data_path + '/stim_key_normalized.csv'): # if we have not already saved a normalized version of the key
            stim_key = pd.read_csv(os.path.join(data_path, 'StimulusLocationInfo.csv')) # get 'key' of stim position coordinates in % form
            # print(f'stim_key: {stim_key.shape}') # print the shape of it for debugging

            stim_key.iloc[:,1] = stim_key.iloc[:,1].str.strip('%').astype(float)/100 # convert X to decimal from percentage
            stim_key.iloc[:,2] = stim_key.iloc[:,2].str.strip('%').astype(float)/100 # convert Y to decimal from percentage

            # Map X and Y over the screen dimensions
            stim_key['XPos'] = (640 * stim_key['XPos']) # absolute x
            stim_key['YPos'] = (480 * stim_key['YPos']) # absolute y
            
            # recenter X and Y at (0,0)
            stim_key['XPos'] = stim_key['XPos'] - 320 # 320 is half of 640
            stim_key['YPos'] = stim_key['YPos'] - 240 # 240 is half of 480

            stim_key["Dist"] = np.sqrt(stim_key['XPos']**2 + stim_key['YPos']**2) # absolute (unnormalized) distance from center
            # stim_key["Angle"] = np.arctan2(stim_key['YPos'], stim_key['XPos']) # angle calculated using absolute (unnormalized) distances
            
            # print(f'stim_key: {stim_key}')

            pd.DataFrame(stim_key).to_csv(data_path + '/stim_key_normalized.csv') # save as csv
        else: # if we already made it before
            stim_key = pd.read_csv(os.path.join(data_path, 'stim_key_normalized.csv')) # load the normalized vers        
        final_subject_list = [] # list to store IDs of only the subjects we end up continuing with (sufficient trials)

        for z, subject in enumerate(subject_list):
            filename = subject_ids_dict[subject]
            print(f'filename: {filename}')

            # below is equivalent to "get_all_unnormalized_data_this_session"
            data = pd.read_csv(os.path.join(data_path, filename))
            data = data.drop('Unnamed: 0', axis=1) # remove first column that just has the indices (it will make it again anyway)

            condition = data['Condition'] # trial type (WM or CTL)

            # output variable to be predicted: 
            resp = data['ProbeDisp.RESP'] - 7 # to encode as 0 and 1 instead of 7 and 8
            # print(np.unique(resp))       
            resp = resp.fillna(-1) # fill nans with -1 for violation, otherwise uncomment the following
            data['Response'] = resp # store modified response column with 1s and 0s and empty spots filled in
            prev_choice, locs_mapping = create_previous_choice_vector(resp)
            del resp # delete it so I don't accidentally try to use it after

            # get normalized coordinates, distances, angles from stimulus positions using key and save as new columns in dataframe
            data = pd.merge(data, stim_key.add_suffix('_stim_probe'), left_on='Stimulus_probe', right_on='Stimulus_stim_probe', how='left')
            data = pd.merge(data, stim_key.add_suffix('_stim_1'), left_on='Stimulus_dot1', right_on='Stimulus_stim_1', how='left')
            data = pd.merge(data, stim_key.add_suffix('_stim_2'), left_on='Stimulus_dot2', right_on='Stimulus_stim_2', how='left')
            data = pd.merge(data, stim_key.add_suffix('_stim_3'), left_on='Stimulus_dot3', right_on='Stimulus_stim_3', how='left')
            data = data.drop(columns=['Stimulus_stim_probe', 'Stimulus_stim_1', 'Stimulus_stim_2', 'Stimulus_stim_3'])

            # distances between dots and probe
            data['1_to_probe_dist'] = np.sqrt((data['XPos_stim_1']-data['XPos_stim_probe'])**2 + (data['YPos_stim_1']-data['YPos_stim_probe'])**2)
            data['2_to_probe_dist'] = np.sqrt((data['XPos_stim_2']-data['XPos_stim_probe'])**2 + (data['YPos_stim_2']-data['YPos_stim_probe'])**2)
            data['3_to_probe_dist'] = np.sqrt((data['XPos_stim_3']-data['XPos_stim_probe'])**2 + (data['YPos_stim_3']-data['YPos_stim_probe'])**2)

            data['min_dist'] = data[['1_to_probe_dist', '2_to_probe_dist', '3_to_probe_dist']].min(axis=1) # yes biased when smaller
            data['avg_dist'] = data[['1_to_probe_dist', '2_to_probe_dist', '3_to_probe_dist']].mean(axis=1) # no biased when larger?

            # prev choice and prev accuracy
            data['prev_resp'] = prev_choice
            # print(data.head())

            pd.DataFrame(data).to_csv(data_path + '/partially_processed/preproc_' + filename) # save data with the new columns added and blanks filled in

            if delete_first_trial: # if the flag to delete the first trial is on
                data = data.iloc[1:] # take only the data from the second trial onward

            if wm_only == True: # get only working memory trials if the condition is set to do so
                data = data.loc[data['Condition']=='WM']
                # print(f'WM only: {data.head()}')
                prev_choice = prev_choice[np.where(data['Condition']=='WM')]
                # prev_accuracy = prev_accuracy[np.where(data['Condition']=='WM')]

            # check to see if there are sufficient trials in this subject's session
            if data['ProbeDisp.RESP'].isna().sum()/len(data['ProbeDisp.RESP']) > 0.2: # if more than 20% of trials are nan
                print(f'Insufficient trials for subject {subject}. Excluding {subject_ids_dict[subject]} from list.')
                continue # skip this subject and move on to the next iteration
            else:
                final_subject_list.append(subject) # if it has enough trials, append it to the new list

            # create_design_mat:            
            unnormalized_inpt = []
            vars_to_keep = [ # save unnormalized distances and angles here so I can scale them all together later
                'Dist_stim_probe', 
                'Dist_stim_1', '1_to_probe_dist',
                'Dist_stim_2', '2_to_probe_dist',
                'Dist_stim_3', '3_to_probe_dist',
                'min_dist', 'avg_dist'
            ]
            # additional_features = [prev_choice, prev_accuracy]
            # unnormalized_inpt = np.column_stack([data[col] for col in vars_to_keep] + additional_features)
            unnormalized_inpt = np.column_stack([data[col] for col in vars_to_keep] + prev_choice)
            # print(unnormalized_inpt)

            y = np.expand_dims(data['Response'], axis=1) # don't need to remap choice vals for our task (?)
            # correct = np.expand_dims(prev_accuracy, axis=1)
            
            # if num_viols_50 < 10 ? what filter should I use here?
            subj_unnormalized_inpt = np.copy(unnormalized_inpt)
            subj_y = np.copy(y)
            # subj_session = 1 # not sure if I can delete this later
            # subj_correct = np.copy(correct)

            final_subject_ids_dict[subject].append(filename)

            # write out subject's unnormalized data matrix:
            np.savez(processed_data_path + 'data_by_subj/' + group_str + '_' + subject + '_unnormalized.npz', subj_unnormalized_inpt, subj_y)
            subj_trial_fold_lookup = split_train_test(subj_unnormalized_inpt, split_ratio=0.7)
            np.savez(processed_data_path + 'data_by_subj/' + group_str + '_' + subject + '_trial_fold_lookup.npz', subj_trial_fold_lookup)
            
            # np.savez(processed_data_path + 'data_by_subj/' + group_str + '_' + subject + '_correct.npz', subj_correct)
            # assert subj_correct.shape[0] == subj_y.shape[0] # ?

            # now create or append data to master array across all subjects:
            if z == 0:
                master_inpt = np.copy(subj_unnormalized_inpt)
                subject_start_idx[subject] = 0
                subject_end_idx[subject] = master_inpt.shape[0] - 1
                master_y = np.copy(subj_y)
                # master_session = subj_session
                master_trial_fold_lookup_table = subj_trial_fold_lookup
                # master_correct = np.copy(subj_correct)
            else:
                subject_start_idx[subject] = master_inpt.shape[0]
                master_inpt = np.vstack((master_inpt, subj_unnormalized_inpt))
                subject_end_idx[subject] = master_inpt.shape[0] - 1
                master_y = np.vstack((master_y, subj_y))
                # master_session = np.concatenate((master_session, subj_session))
                master_trial_fold_lookup_table = np.vstack(
                    (master_trial_fold_lookup_table, subj_trial_fold_lookup))
                # master_correct = np.vstack((master_correct, subj_correct))
            
        # num_subjects = len(final_subject_list)

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

        # normalization happens here
        normalized_inpt = np.copy(master_inpt) 
        # print(f'size normalized_inpt before scale: {np.shape(normalized_inpt)}')
        print(f'means before scaling: {normalized_inpt.mean(axis=0)}, sds before scaling: {normalized_inpt.std(axis=0)}')
        normalized_inpt[:, :-2] = preprocessing.scale(normalized_inpt[:, :-2]) # scale all features except the last two cols (prev choice and prev acc)
        # print(f'size normalized_inpt after scale: {np.shape(normalized_inpt)}')
        print(f'means after scaling: {normalized_inpt.mean(axis=0)}, sds after scaling: {normalized_inpt.std(axis=0)}')

        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_inpt[:, :-2] = min_max_scaler.fit_transform(normalized_inpt[:, :-2])
        print(f'means after min max scaling: {normalized_inpt.mean(axis=0)}, sds after min max scaling: {normalized_inpt.std(axis=0)}')

        np.savez(processed_data_path + group_str + '_all_subj_concat.npz',
            normalized_inpt,
            master_y)
        pd.DataFrame(normalized_inpt).to_csv(processed_data_path + group_str + '_all_subj_concat.csv') # also save as csv for readability
        pd.DataFrame(master_y).to_csv(processed_data_path + group_str + '_all_subj_concat_y.csv') # also save master_y as csv
        np.savez(
            processed_data_path + group_str + '_all_subj_concat_unnormalized.npz',
            master_inpt, master_y)
        np.savez(
            processed_data_path + group_str + '_all_subj_concat_trial_fold_lookup.npz',
            master_trial_fold_lookup_table)
        pd.DataFrame(master_trial_fold_lookup_table).to_csv(processed_data_path + group_str + '_all_subj_concat_trial_fold_lookup.csv') # save as csv for readability
        # np.savez(processed_data_path + group_str + '_all_subj_concat_correct.npz',
        #     master_correct)
        np.savez(processed_data_path + 'data_by_subj/' + group_str + '_final_subject_list.npz',
            final_subject_list) # ?
        
        # json = json.dumps(final_subject_ids_dict)
        # f = open(processed_data_path + 'final_subject_ids_dict.json', "w")
        # f.write(json)
        # f.close()

        # Now write out normalized data (when normalized across all subjects) for
        # each subject:        
        counter = 0
        for subject in subject_start_idx.keys():
            start_idx = subject_start_idx[subject]
            end_idx = subject_end_idx[subject]
            inpt = normalized_inpt[range(start_idx, end_idx + 1)]
            y = master_y[range(start_idx, end_idx + 1)]
            # session = master_session[range(start_idx, end_idx + 1)]
            counter += inpt.shape[0]
            np.savez(processed_data_path + 'data_by_subj/' + group_str + '_' + subject + '_processed.npz',
                inpt, y)

        assert counter == master_inpt.shape[0]

        reaction_time = data['TrialRT']
        if not os.path.exists('response_times/data_by_subj/'):
            os.makedirs('response_times/data_by_subj/')
        np.savez(data_path + 'response_times/data_by_subj/' + group_str + '_' + subject + '.npz', reaction_time)

        final_subject_list_all[group] = [subj for subj in final_subject_list]
        final_subj_list_len[group] = len(final_subject_list_all[group]) # store length of final subject list for this group
    
    print(f'Group {1}: {final_subj_list_len[1]} of {orig_subj_list_len[1]} subjects, {orig_subj_list_len[1]-final_subj_list_len[1]} removed, {final_subj_list_len[1]/orig_subj_list_len[1]*100}% kept')
    print(f'Group {2}: {final_subj_list_len[2]} of {orig_subj_list_len[2]} subjects, {orig_subj_list_len[2]-final_subj_list_len[2]} removed, {final_subj_list_len[2]/orig_subj_list_len[2]*100}% kept')
    print(f'Group {3}: {final_subj_list_len[3]} of {orig_subj_list_len[3]} subjects, {orig_subj_list_len[3]-final_subj_list_len[3]} removed, {final_subj_list_len[3]/orig_subj_list_len[3]*100}% kept')

    print(final_subject_list_all)
    # save list of all final subjects as dictionary with groups: subj ids
    json = json.dumps(final_subject_list_all)
    f = open(processed_data_path + 'data_by_subj/' + 'total_final_subject_list.json', "w")
    f.write(json)
    f.close()