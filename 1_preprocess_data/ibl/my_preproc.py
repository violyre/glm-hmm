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

# npr.seed(65)
delete_first_trial = True # change this flag if you don't want to delete the first trial

if __name__ == '__main__':
    data_path = 'C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/'
    # create directory for saving data:
    if not os.path.exists(data_path + "partially_processed/"):
        os.makedirs(data_path + "partially_processed/")

    os.chdir(data_path) # search in correct directory
    final_subject_list_all = {} # empty dictionary to store all the subj from each group that we keep

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
                    # group = match.group(1) # group number
                    # subject = match.group(2) # ID number
                    # chronicity = match.group(3) # chronicity -- ignoring this for now?
                    subject = match.group(1) # ID number 

                    if subject not in subject_list:
                        subject_list.append(subject)

                    # print(f'Identified filename: {filename}, subject {subject} for group 1')

                    # file_path = os.path.join(data_path, filename) # full path to each CSV file
                    # df = pd.read_csv(file_path) # read the CSV file 
                    # subject_ids_dict[subject] = df # store it in the dataframe

                    subject_ids_dict[subject] = filename # store filename for subject
                    # subject_ids_dict[subject].append(filename) # append filename -- only makes sense if there are multiple files per subject
                    # subject_ids_dict[group][subject].append(filename)
        # print(f'subject_list: {orig_subject_list}')
        # print(f'subject_ids_dict: {subject_ids_dict[subject]}, subject {subject}')
        np.savez('partially_processed/subject_list_' + group_str + '.npz', subject_list)

        # Create directories for saving data:
        processed_data_path = data_path + "data_for_cluster/"
        if not os.path.exists(processed_data_path):
            os.makedirs(processed_data_path)
        # Also create a subdirectory for storing each individual subject's data:
        if not os.path.exists(processed_data_path + "data_by_subj/"):
            os.makedirs(processed_data_path + "data_by_subj/")

        wm_only = True # change to False if you want to store all trial types, including CTL

        # make sure each subject has responses for at least 80% of trials
        # subject_list = [] # store only subject IDs that have enough trials 
        # for subject in orig_subject_list:
        #     filename = subject_ids_dict[subject]
        #     data = pd.read_csv(os.path.join(data_path, filename))
        #     if wm_only == True:
        #         data = data.loc[data['Condition']=='WM'] # restrict data to WM only 

        #     if data['ProbeDisp.RESP'].isna().sum()/len(data['ProbeDisp.RESP']) > 0.2: # if more than 20% of trials are nan
        #         print(f'Insufficient trials for subject {subject}. Excluding {subject_ids_dict[subject]} from list.')
        #     else:
        #         subject_list.append(subject) # if it has enough trials, append it to the new list
        # # subject_list = [subject for subject in subject_list 
        # #                 if pd.read_csv(os.path.join(data_path, subject_ids_dict[subject]))['ProbeDisp.RESP'].isna().sum() / 
        # #                 len(pd.read_csv(os.path.join(data_path, subject_ids_dict[subject]))['ProbeDisp.RESP']) <= 0.2]
        # num_subjects = len(subject_list)
        # print(f'Number of subjects with sufficient number of trials: {num_subjects}')
        # print(f'Subject list: {subject_list}')

        # Identify idx in master array where each subject's data starts and ends:
        subject_start_idx = {}
        subject_end_idx = {}

        final_subject_ids_dict = defaultdict(list) # should I do {}?
        # WORKHORSE: iterate through each animal and each animal's set of eids;
        # obtain unnormalized data.  Write out each animal's data and then also
        # write to master array

        if not os.path.exists(data_path + '/stim_key_normalized.csv'): # if we have not already saved a normalized version of the key
            stim_key = pd.read_csv(os.path.join(data_path, 'StimulusLocationInfo.csv')) # get 'key' of stim position coordinates in % form
            # print(f'stim_key: {stim_key.shape}') # print the shape of it for debugging

            stim_key.iloc[:,1] = stim_key.iloc[:,1].str.strip('%').astype(float)/100 # convert X to decimal from percentage
            stim_key.iloc[:,2] = stim_key.iloc[:,2].str.strip('%').astype(float)/100 # convert Y to decimal from percentage
            stim_key['XPos'] = (2 * stim_key['XPos']) - 1 # map X over range [-1, 1]
            stim_key['YPos'] = (2 * stim_key['YPos']) - 1 # map Y over range [-1, 1]

            #stim_key["XY"] = stim_key['XPos'] * stim_key['YPos']
            stim_key["Dist"] = np.sqrt(stim_key['XPos']**2 + stim_key['YPos']**2)
            stim_key["Angle"] = np.arctan(stim_key['YPos']/stim_key['XPos'])
            # print(f'stim_key: {stim_key}')

            pd.DataFrame(stim_key).to_csv(data_path + '/stim_key_normalized.csv') # save as csv
        else: # if we already made it before
            stim_key = pd.read_csv(os.path.join(data_path, 'stim_key_normalized.csv')) # load the normalized vers        
        final_subject_list = [] # list to store IDs of only the subjects we end up continuing with (sufficient trials)

        for z, subject in enumerate(subject_list):
            # sess_counter = 0
            ###
            # for filename in subject_ids_dict[subject].values():
            filename = subject_ids_dict[subject]
            # print(f'subject_ids_dict[{subject}]: {subject_ids_dict[subject]}')
            print(f'filename: {filename}')

            # below is equivalent to "get_all_unnormalized_data_this_session"
            # create unnormalized input with first col stim_probe, next three cols x3 stim_1, stim_2, stim_3 (X, Y, and XY)
            # then past choice, then past reward
            data = pd.read_csv(os.path.join(data_path, filename))
            data = data.drop('Unnamed: 0', axis=1) # remove first column that just has the indices (it will make it again anyway)

            condition = data['Condition'] # trial type (WM or CTL)

            # output variable to be predicted: 
            resp = data['ProbeDisp.RESP'] - 7 # to encode as 0 and 1 instead of 7 and 8
            # print(np.unique(resp))       
            resp = resp.fillna(-1) # fill nans with -1 for violation, otherwise uncomment the following
            # for idx, val in enumerate(resp):
            #     if np.isnan(val):
                    # # fill in incorrect choice for any nans
                    # if condition[idx] == 'CTL': 
                    #     resp[idx] = 1
                    # else: # if WM, fill in nan with incorrect response
                    #     if data['ProbeDisp.CRESP'].iloc[idx] - 7 == 0: 
                    #         resp[idx] = 1 
                    #     else:
                    #         resp[idx] = 0
                    # # resp[idx] = lambda idx: 1 if condition[idx] == 'CTL' # unfinished list comprehension attempt
            # print(np.unique(resp))
            # print(data.head())

            # input variables:
            accuracy = data['TrialAccuracy']

            data['Response'] = resp # store modified response column with 1s and 0s and empty spots filled in

            # data['Prev_Response'] = resp.shift(1).fillna(0) # previous trial's response
            # data['Prev_Accuracy'] = data['TrialAccuracy'].shift(1).fillna(0) # previous trial's accuracy
            prev_choice, locs_mapping = create_previous_choice_vector(resp)
            # modified create_wsls_covariate():
            prev_accuracy = np.hstack([np.array(accuracy[0]), accuracy])[:-1]
            # Now need to go through and update previous reward to correspond to
            # same trial as previous choice:
            for i, loc in enumerate(locs_mapping[:, 0]):
                nearest_loc = locs_mapping[i, 1]
                prev_accuracy[loc] = prev_accuracy[nearest_loc]

            del resp # delete it so I don't accidentally try to use it after

            # get normalized coordinates from stimulus positions using key and save as new columns in dataframe
            data = pd.merge(data, stim_key.add_suffix('_stim_probe'), left_on='Stimulus_probe', right_on='Stimulus_stim_probe', how='left')
            data = pd.merge(data, stim_key.add_suffix('_stim_1'), left_on='Stimulus_dot1', right_on='Stimulus_stim_1', how='left')
            data = pd.merge(data, stim_key.add_suffix('_stim_2'), left_on='Stimulus_dot2', right_on='Stimulus_stim_2', how='left')
            data = pd.merge(data, stim_key.add_suffix('_stim_3'), left_on='Stimulus_dot3', right_on='Stimulus_stim_3', how='left')
            data = data.drop(columns=['Stimulus_stim_probe', 'Stimulus_stim_1', 'Stimulus_stim_2', 'Stimulus_stim_3'])
            data['prev_resp'] = prev_choice
            data['prev_acc'] = prev_accuracy
            # print(data.head())

            pd.DataFrame(data).to_csv(data_path + '/partially_processed/preproc_' + filename) # save data with the new columns added and blanks filled in

            if delete_first_trial: # if the flag to delete the first trial is on
                data = data.iloc[1:] # take only the data from the second trial onward

            if wm_only == True: # get only working memory trials if the condition is set to do so
                data = data.loc[data['Condition']=='WM']
                # print(f'WM only: {data.head()}')
                prev_choice = prev_choice[np.where(data['Condition']=='WM')]
                prev_accuracy = prev_accuracy[np.where(data['Condition']=='WM')]

            # check to see if there are sufficient trials in this subject's session
            if data['ProbeDisp.RESP'].isna().sum()/len(data['ProbeDisp.RESP']) > 0.2: # if more than 20% of trials are nan
                print(f'Insufficient trials for subject {subject}. Excluding {subject_ids_dict[subject]} from list.')
                continue # skip this subject and move on to the next iteration
            else:
                final_subject_list.append(subject) # if it has enough trials, append it to the new list

            # create_design_mat:
            unnormalized_inpt = np.zeros((len(data['Condition']), 18)) # change number of weights here

            unnormalized_inpt[:,0] = data['XPos_stim_probe']
            unnormalized_inpt[:,1] = data['YPos_stim_probe']
            unnormalized_inpt[:,2] = data['Dist_stim_probe']
            unnormalized_inpt[:,3] = data['Angle_stim_probe']

            unnormalized_inpt[:,4] = data['XPos_stim_1']
            unnormalized_inpt[:,5] = data['YPos_stim_1']
            unnormalized_inpt[:,6] = data['Dist_stim_1']
            unnormalized_inpt[:,7] = data['Angle_stim_1']

            unnormalized_inpt[:,8] = data['XPos_stim_2']
            unnormalized_inpt[:,9] = data['YPos_stim_2']
            unnormalized_inpt[:,10] = data['Dist_stim_2']
            unnormalized_inpt[:,11] = data['Angle_stim_2']

            unnormalized_inpt[:,12] = data['XPos_stim_3']
            unnormalized_inpt[:,13] = data['YPos_stim_3']
            unnormalized_inpt[:,14] = data['Dist_stim_3']
            unnormalized_inpt[:,15] = data['Angle_stim_3']

            unnormalized_inpt[:,16] = prev_choice
            unnormalized_inpt[:,17] = prev_accuracy

            y = np.expand_dims(data['Response'], axis=1) # don't need to remap choice vals for our task (?)
            correct = np.expand_dims(prev_accuracy, axis=1)
            
            # if num_viols_50 < 10 ? what filter should I use here?
            subj_unnormalized_inpt = np.copy(unnormalized_inpt)
            subj_y = np.copy(y)
            # subj_session = 1 # not sure if I can delete this later
            subj_correct = np.copy(correct)

            final_subject_ids_dict[subject].append(filename)

            # write out subject's unnormalized data matrix:
            np.savez(processed_data_path + 'data_by_subj/' + group_str + '_' + subject + '_unnormalized.npz', subj_unnormalized_inpt, subj_y)
            # trial_fold_lookup = 
            np.savez(processed_data_path + 'data_by_subj/' + group_str + '_trial_fold_lookup.npz', trial_fold_lookup)
            
            np.savez(processed_data_path + 'data_by_subj/' + group_str + '_' + subject + '_correct.npz', subj_correct)
            assert subj_correct.shape[0] == subj_y.shape[0] # ?

            # now create or append data to master array across all subjects:
            if z == 0:
                master_inpt = np.copy(subj_unnormalized_inpt)
                subject_start_idx[subject] = 0
                subject_end_idx[subject] = master_inpt.shape[0] - 1
                master_y = np.copy(subj_y)
                # master_session = subj_session
                master_correct = np.copy(subj_correct)
            else:
                subject_start_idx[subject] = master_inpt.shape[0]
                master_inpt = np.vstack((master_inpt, subj_unnormalized_inpt))
                subject_end_idx[subject] = master_inpt.shape[0] - 1
                master_y = np.vstack((master_y, subj_y))
                # master_session = np.concatenate((master_session, subj_session))
                master_correct = np.vstack((master_correct, subj_correct))
            
        # num_subjects = len(final_subject_list)

        ####
        # write out data from across subjects
        assert np.shape(master_inpt)[0] == np.shape(master_y)[
            0], "inpt and y not same length"
        assert np.shape(master_correct)[0] == np.shape(master_y)[
            0], "correct and y not same length"
        # assert len(subject_list) == num_subjects, f"{num_subjects} subjects in group 1" # not sure what the point of doing this for us is

        normalized_inpt = np.copy(master_inpt)
        # print(f'size normalized_inpt before scale: {np.shape(normalized_inpt)}')
        # normalized_inpt[:, 0] = preprocessing.scale(normalized_inpt[:, 0]) # ?
        # print(f'size normalized_inpt after scale: {np.shape(normalized_inpt)}')
        np.savez(processed_data_path + group_str + '_all_subj_concat.npz',
            normalized_inpt,
            master_y)
        # np.savetxt(processed_data_path + 'all_subj_concat' + '.csv',
        #     normalized_inpt)
        pd.DataFrame(normalized_inpt).to_csv(processed_data_path + group_str + '_all_subj_concat.csv') # also save as csv for readability
        pd.DataFrame(master_y).to_csv(processed_data_path + group_str + '_all_subj_concat_y.csv') # also save master_y as csv
        np.savez(
            processed_data_path + group_str + '_all_subj_concat_unnormalized.npz',
            master_inpt, master_y)
        np.savez(processed_data_path + group_str + '_all_subj_concat_correct.npz',
            master_correct)
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
    print(final_subject_list_all)
    # save list of all final subjects as dictionary with groups: subj ids
    json = json.dumps(final_subject_list_all)
    f = open(processed_data_path + 'data_by_subj/' + 'total_final_subject_list.json', "w")
    f.write(json)
    f.close()