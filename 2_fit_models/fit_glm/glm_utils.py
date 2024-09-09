import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from GLM import glm
from scipy.stats import chi2 # my addition

npr.seed(65)


def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    session = data[2]
    return inpt, y, session


def fit_glm(inputs, datas, M, C):
    new_glm = glm(M, C)
    new_glm.fit_glm(datas, inputs, masks=None, tags=None)
    # print(f'Wk: {new_glm.Wk}') # my addition
    # print(f'params: {new_glm.params}') # my addition
    # Get loglikelihood of training data:
    loglikelihood_train = new_glm.log_marginal(datas, inputs, None, None)
    # deviance = -2 * loglikelihood_train # my addition
    # print(f'deviance: {deviance}')
    recovered_weights = new_glm.Wk
    return loglikelihood_train, recovered_weights


# Append column of zeros to weights matrix in appropriate location
def append_zeros(weights):
    weights_tranpose = np.transpose(weights, (1, 0, 2))
    weights = np.transpose(
        np.vstack([
            weights_tranpose,
            np.zeros((1, weights_tranpose.shape[1], weights_tranpose.shape[2]))
        ]), (1, 0, 2))
    return weights


def load_session_fold_lookup(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    session_fold_lookup_table = data[0]
    return session_fold_lookup_table


def load_animal_list(list_file):
    container = np.load(list_file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_list = data[0]
    return animal_list

def load_subj_list(list_file):
    container = np.load(list_file, allow_pickle=True)
    data = [container[key] for key in container]
    subj_list = data[0]
    return subj_list

def plot_input_vectors(Ws,
                    #    p_values, # my addition
                       figure_directory,
                       title='true',
                       save_title="true",
                       labels_for_plot=[]):
    K = Ws.shape[0]
    K_prime = Ws.shape[1]
    M = Ws.shape[2] - 1
    # print(f'K: {K}, K_prime: {K_prime}, M: {M}') # my addition
    # print(f'Ws: {Ws}, shape {Ws.shape}') # my addition
    
    fig = plt.figure(figsize=(7, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.15,
                        bottom=0.27,
                        right=0.95,
                        top=0.95,
                        wspace=0.3,
                        hspace=0.3)

    for j in range(K):
        for k in range(K_prime - 1):
            # plt.subplot(K, K_prime, 1+j*K_prime+k)
            plt.plot(range(M + 1), -Ws[j][k], marker='o')
            plt.plot(range(-1, M + 2), np.repeat(0, M + 3), 'k', alpha=0.2)
            plt.axhline(y=0, color="k", alpha=0.5, ls="--")
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
                
            # # plot p_value annotations: my addition
            # for i in range(M+1):
            #     p_val = p_values[j][k][i]
            #     if p_val < 0.001:
            #         significance = '***'
            #     elif p_val < 0.01:
            #         significance = '**'
            #     elif p_val < 0.05:
            #         significance = '*'
            #     else:
            #         significance = ''

            #     if significance:
            #         plt.annotate(significance, (i, -Ws[j][k][i]), textcoords="offset points", xytext=(0,10), ha='center')

            #plt.ylim((-3, 6))
            plt.ylim((-6,6))

    fig.text(0.04,
             0.5,
             "Weight",
             ha="center",
             va="center",
             rotation=90,
             fontsize=15)
    fig.suptitle("GLM Weights: " + title, y=0.99, fontsize=14)
    fig.savefig(figure_directory + 'glm_weights_' + save_title + '.png')

    # fig.show() # my addition

# plot log-likelihood comparisons between removing features and original
def plot_feature_selection_ll(all_labels,loglikelihood_vectors,original_loglikelihood,num_folds,
                                directory,
                                title,
                                save_title,
                                type):
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.xlabel('Removed Feature')
    plt.xticks(rotation=90)
    plt.tight_layout()
    if type == 'Fold':
        plt.ylabel('Log-Likelihood')
        for fold in range(num_folds):
            plt.plot(all_labels,loglikelihood_vectors[fold], marker='o', label=f'Fold {fold}')
        plt.axhline(y=original_loglikelihood, color='r', linestyle='--', label='Original Log-Likelihood')
    elif type == 'GroupDiff':
        plt.ylabel('Difference in Log-Likelihood')
        for group in range(1,4):
            plt.plot(all_labels, loglikelihood_vectors[group-1]-original_loglikelihood[group-1], marker='o', label=f'Group {group}')
        plt.axhline(y=0, color='r', linestyle='--', label='No Difference')
    elif type == 'Group':
        plt.ylabel('Log-Likelihood')
        for group in range(1,4):
            plt.plot(all_labels, loglikelihood_vectors[group-1], marker='o', label=f'Group {group}')
            plt.axhline(y=original_loglikelihood[group-1], color='r', linestyle='--', label=f'Original Log-Likelihood, Group {group}')
    else: 
        raise Exception("Invalid type! Must be 'Fold', 'Group', or 'GroupDiff'")
    plt.title(f'Feature Selection Log-Likelihood Comparison for {title}')
    plt.legend()
    plt.savefig(f"{directory}{save_title}")
    #plt.show()

# update the feature list based on features to remove
def update_features(features_to_remove, all_labels):
    # Filter out the features to remove
    # features_to_keep = [label for label in all_labels if label not in features_to_remove]
    feat_idxs_to_keep = [idx for idx, feat in enumerate(all_labels) if feat not in features_to_remove]
    # features_to_keep = [all_labels[i] for i in feat_idxs_to_keep]
    return feat_idxs_to_keep