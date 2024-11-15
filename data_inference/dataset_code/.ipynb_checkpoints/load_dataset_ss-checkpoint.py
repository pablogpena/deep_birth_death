# +
import random
from datetime import datetime

from dataset_code.utilities_ss import *
from dataset_code.utilities_vec import *

import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
# -

val_perc = 0.1


# # Dataset class

class Dataset:
    def __init__(self):
        self.X_ss = []
        
        self.label = []
        self.label_names = []
        
        self.reg_values = []
        
        self.time = []
        self.frac0 = []
        self.frac1 = []
        
        self.lambda0 = []
        self.lambda1 = []
        self.mu0 = []
        self.mu1 = []
        
        self.r0 = []
        self.r1 = []
        self.a0 = []
        self.a1 = []
        
        self.norm_r0 = []
        self.norm_r1 = []
        self.norm_a0 = []
        self.norm_a1 = []
        self.norm_time = []
        self.norm_frac0 = []
        self.norm_frac1 = []
        self.norm_lambda0 = []
        self.norm_lambda1 = []
        self.norm_mu0 = []
        self.norm_mu1 = []
        
        self.resc_factor = []


# # Complete dataset

def load_dataset(path_list):
    X_ss = []
    
    # Classification data
    label = []
    label_names = []
    
    # Regression data
    r0 = []
    r1 = []
    a0 = []
    a1 = []
    mu0 = []
    mu1 = []
    lambda0 = []
    lambda1 = []
    time = []
    frac0 = []
    frac1 = []
    resc_factor = []

    norm_r0 = []
    norm_r1 = []
    norm_a0 = []
    norm_a1 = []
    norm_mu0 = []
    norm_mu1 = []
    norm_lambda0 = []
    norm_lambda1 = []
    norm_time = []
    norm_frac0 = []
    norm_frac1 = []

    # Each path must correspond to a different class
    for i, path in enumerate(path_list):

        # Load tree from path
        df = pd.read_csv(path, sep="|")

        print(datetime.now().strftime("%H:%M:%S"), ". Loading trees from: ", path)        
        
        # Generate summary statistics from tree
        print("Encoding SS trees")
        tree_ss = load_tree_ss(path)
        X_ss.extend(tree_ss.to_numpy())
        
        # Save regression data from tree
        r0.extend(df['r0'].to_numpy())
        r1.extend(df['r1'].to_numpy())
        a0.extend(df['a0'].to_numpy())
        a1.extend(df['a1'].to_numpy())
        mu0.extend(df['mu0'].to_numpy())
        mu1.extend(df['mu1'].to_numpy())
        lambda0.extend(df['lambda0'].to_numpy())
        lambda1.extend(df['lambda1'].to_numpy())
        time.extend(df['time'].to_numpy())
        frac0.extend(df['frac0'].to_numpy())
        frac1.extend(df['frac1'].to_numpy())
        resc_factor.extend(df['resc_factor'].to_numpy())
        
        norm_r0.extend(df['norm_r0'].to_numpy())
        norm_r1.extend(df['norm_r1'].to_numpy())
        norm_a0.extend(df['norm_a0'].to_numpy())
        norm_a1.extend(df['norm_a1'].to_numpy())
        norm_mu0.extend(df['norm_mu0'].to_numpy())
        norm_mu1.extend(df['norm_mu1'].to_numpy())
        norm_lambda0.extend(df['norm_lambda0'].to_numpy())
        norm_lambda1.extend(df['norm_lambda1'].to_numpy())
        norm_time.extend(df['norm_time'].to_numpy())
        norm_frac0.extend(df['norm_frac0'].to_numpy())
        norm_frac1.extend(df['norm_frac1'].to_numpy())
    
        # Save label from tree
        label.extend(np.zeros(len(tree_vector)) + i)
        label_names.append(path)
        
    # generate Dataset object
    dataset = Dataset()
    dataset.X_ss = np.asarray(X_ss).astype('float32')
    
    dataset.label = label
    dataset.label_names = label_names
    
    dataset.r0 = r0
    dataset.r1 = r1
    dataset.a0 = a0
    dataset.a1 = a1
    dataset.mu0 = mu0
    dataset.mu1 = mu1
    dataset.lambda0 = lambda0
    dataset.lambda1 = lambda1
    dataset.time = time
    dataset.frac0 = frac0
    dataset.frac1 = frac1
    dataset.resc_factor = resc_factor
    
    dataset.norm_r0 = norm_r0
    dataset.norm_r1 = norm_r1
    dataset.norm_a0 = norm_a0
    dataset.norm_a1 = norm_a1
    dataset.norm_mu0 = norm_mu0
    dataset.norm_mu1 = norm_mu1
    dataset.norm_lambda0 = norm_lambda0
    dataset.norm_lambda1 = norm_lambda1
    dataset.norm_time = norm_time
    dataset.norm_frac0 = norm_frac0
    dataset.norm_frac1 = norm_frac1
        
    return dataset


# # Utils

def shuffle_two_arrays(a, b):
    combined = list(zip(a, b))
    random.shuffle(combined)
    a_permuted, b_permuted = zip(*combined)
    
    return np.array(a_permuted), np.array(b_permuted)


def shuffle_three_arrays(a, b, c):
    combined = list(zip(a, b, c))
    random.shuffle(combined)
    a_permuted, b_permuted, c_permuted = zip(*combined)
    
    return np.array(a_permuted), np.array(b_permuted), np.array(c_permuted)


def shuffle_four_arrays(a, b, c, d):
    combined = list(zip(a, b, c, d))
    random.shuffle(combined)
    a_permuted, b_permuted, c_permuted, d_permuted = zip(*combined)
    
    return np.array(a_permuted), np.array(b_permuted), np.array(c_permuted), np.array(d_permuted)


def shuffle_five_arrays(a, b, c, d, e):
    combined = list(zip(a, b, c, d, e))
    random.shuffle(combined)
    a_permuted, b_permuted, c_permuted, d_permuted, e_permuted = zip(*combined)
    
    return np.array(a_permuted), np.array(b_permuted), np.array(c_permuted), np.array(d_permuted), np.array(e_permuted)
