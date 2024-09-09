import autograd.numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    stim_key_norm = pd.read_csv('C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/stim_key_normalized.csv')
    print(stim_key_norm.head())
    stim_key = pd.read_csv('C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/StimulusLocationInfo.csv')
    print(stim_key.head())

    fig = plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.xlim((-1,1))
    plt.ylim((-1,1))

    plt.plot(0,0,marker='+',color='black')
    plt.scatter(stim_key_norm['XPos'],stim_key_norm['YPos'],marker='o',color='red')
    plt.title('Normalized Dot Positions')
    plt.grid()
    plt.show()
    fig.savefig('C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/dot_positions_normalized.png')

    fig = plt.figure(figsize=(7,7), dpi=80, facecolor='w', edgecolor='k')
    plt.xlim((0,100))
    plt.ylim((0,100))

    stim_key.iloc[:,1] = stim_key.iloc[:,1].str.strip('%').astype(float) # convert X to decimal from percentage
    stim_key.iloc[:,2] = stim_key.iloc[:,2].str.strip('%').astype(float) # convert Y to decimal from percentage
    
    plt.plot(50,50,marker="+",color='black')
    plt.scatter(stim_key['XPos'],stim_key['YPos'],marker='o',color='red')
    plt.title('Original Dot Positions')
    plt.grid()
    plt.show()
    fig.savefig('C:/Users/violy/Documents/~PhD/Lab/SC/TCP_data/dot_positions_original.png')