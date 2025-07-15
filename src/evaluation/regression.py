# -*- coding: utf-8 -*-
# +
import os, pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from time import time

import seaborn as sns
import matplotlib.pyplot as plt


# -

# # Results structure
# * 674
#  * BD
#    * no_norm
#      * MAE
#      * MAE_norm
#      
#      * n_params
#      * train_time
#      * history
#    * norm
#      * MAE   
#      * MAE_norm
#
#      * n_params
#      * train_time
#      * history
#    
#  * HE
#    * no_norm
#      * ...
#    * norm
#  * ME
#  * SR
#  * SAT
#  
# * 489
#  * BD
#  * ...
#  * SAT
#  
# * 87
#  * ...

# # Generate results

def generate_reg_results(model_path, X, y, y_norm, div_info, resc_factor, scenario, label, norm):
    results = dict()
    
    nn_model = load_model(model_path + 'model.keras')
            
    with open(model_path + 'model_data.pkl', 'rb') as f:
        n_params, train_time = pickle.load(f)
    with open(model_path + 'history.pkl', 'rb') as f:
        history = pickle.load(f)

    # Get regression values of the corresponding scenario
    X = X[div_info == label]
    if norm != 'no_norm':
        y = y_norm[div_info == label]
    else:
        y = y[div_info == label]
    y = [np.array(elem) for elem in y]            
    
    ### Predictions
    start_time = time()
    y_pred = nn_model.predict(X, verbose=0)
    ex_time = time() - start_time
    print("--- Inference time: ", scenario, "scenario",
          "&", norm, ex_time, "seconds ---")           

    ### Get real values for MAEs 
    y = np.array(y)

    ### Desrescale predicted values for each rate in norm trained models
    if norm != 'no_norm':
        resc_factor_test = resc_factor[div_info == label]
        if scenario == "BD" or scenario == "HE" or scenario == "SAT":
            param_idx = [0]
        elif scenario == "ME":
            param_idx = [0, 2]
        elif scenario == "SR" or scenario == "WW":
            param_idx = [0, 1, 4]  
        # Get real r and T 
        y_pred = [[y / r if i in param_idx else y for i, y in enumerate(sublist)] \
                  for sublist, r in zip(y_pred, resc_factor)]

    # Get lambda and mu for each model  
    y_pred = np.array(y_pred)
    y_pred = _extend_y_values(y_pred, scenario)  
    y = _extend_y_values(y, scenario)

    #Get MAE and MAE_norm for our models model and save them
    results['MAE'] = _get_MAE(y_pred, y)
    results['MAE_norm'] = _get_MAE_norm(y_pred, y,
                                       resc_factor=resc_factor,
                                       scenario= scenario)
    return results, ex_time


# # Parameters values y_pred and y_test
# * BD, HE
#  * r, a, lambda, mu
# * ME
#  * r, a, time, frac, lambda, mu
# * SR
#  * r0, r1, a0, a1, time, lambda0, lambda1, mu0, mu1
# * SAT
#  * lambda
# * WW
#  * r0, r1, a0, a1, time, lambda0, lambda1, mu0, mu1

def _extend_y_values(y_list, scenario):
    if scenario=="BD" or scenario=="HE" or scenario=="ME":
        mu = (y_list[:, 1]*y_list[:, 0])/(1-y_list[:, 1])
        lamb = y_list[:, 0] + mu
        
        y_list = np.column_stack((y_list, lamb, mu))
    
    elif scenario== "SR" or scenario== "WW":
        mu_0 = (y_list[:, 2]*y_list[:, 0])/(1-y_list[:, 2])
        mu_1 = (y_list[:, 3]*y_list[:, 1])/(1-y_list[:, 3])

        lamb_0 = y_list[:, 0] + mu_0
        lamb_1 = y_list[:, 1] + mu_1
        
        y_list = np.column_stack((y_list, lamb_0, lamb_1, mu_0, mu_1))
        
    return y_list


# # Regression metrics

def _get_MAE(y_pred, y_real):
    total = 0  
    for y_pred_i, y_real_i in zip(y_pred, y_real):
        total += abs(y_pred_i - y_real_i)
    return total/len(y_pred)


def _get_MAE_norm(y_pred, y_real, resc_factor, scenario):
    total = 0 
    
    if scenario == "BD" or scenario == "HE":
        param_idx = [0, 2, 3] #we also reescale lambda and mu 
    elif scenario == "SAT":
        param_idx = [0] 
    elif scenario == "ME":
        param_idx = [0, 2, 4, 5] #we also reescale lambda and mu
    elif scenario == "SR" or scenario == "WW":
        param_idx = [0, 1, 4, 5, 6, 7, 8]  #we also reescale lambda and mu
    
    #Reescale baack np. arrays transforms into list
    y_pred = [[y * r if i in param_idx else y for i, y in enumerate(sublist)] for sublist, r in zip(y_pred, resc_factor)]  
    y_real = [[y * r if i in param_idx else y for i, y in enumerate(sublist)] for sublist, r in zip(y_real, resc_factor)]

    #Get back to np.array
    y_pred = np.array(y_pred)
    y_real = np.array(y_real)
    
    #Get MAE 
    for y_pred_i, y_real_i in zip(y_pred, y_real):
            total += abs(y_pred_i - y_real_i)
            
    return total/len(y_pred)


# # Regression metrics

def get_regression_norm_results(results, n_tips, scenarios, norm):
    col = ['norm', 'no_norm']
    idx = ['MAE', 'MAE_norm']

    df = pd.DataFrame(columns=col, index=idx)

    for norm in col:
        for metric in idx:
            total = 0
            for scenario in scenarios:
                total += np.mean(results[n_tips][scenario][norm][metric])
            mean = total/len(scenarios)

            df.at[metric, norm] = mean

    return df


def get_regression_div_results(results, n_tips, scenario, norm):
    reg_values = {
                  'BD':['r', 'a', 'lambda', 'mu'],
                  'HE':['r', 'a', 'lambda', 'mu'],
                  'ME':['r', 'a', 'time', 'frac', 'lambda', 'mu'],
                  'SR':['r0', 'r1', 'a0', 'a1', 'time', 'lambda0', 'lambda1', 'mu0', 'mu1'],
                  'WW':['r0', 'r1', 'a0', 'a1', 'time', 'lambda0', 'lambda1', 'mu0', 'mu1'],
                  'SAT':['lambda 0'],
    }
    
    idx = ['MAE', 'MAE_norm']

    df = pd.DataFrame(columns=reg_values[scenario], index=idx)

    for i, param in enumerate(reg_values[scenario]):
        for metric in idx:
            df.at[metric, param] = results[n_tips][scenario][norm][metric][i]

    return df


def new_get_regression_div_results(results, n_tips, scenario, norm, error):
    reg_values = {
        'BD': ['r', 'a'],
        'HE': ['r', 'a'],
        'ME': ['r', 'a', 'time', 'frac'],
        'SR': ['r0', 'r1', 'a0', 'a1', 'time'],
        'WW': ['r0', 'r1', 'a0', 'a1', 'time'],
        'SAT': ['lambda 0'],
    }
    
    values = results[n_tips][scenario][norm][error]

    df = pd.DataFrame([values], columns=reg_values[scenario], index=['MAE'])

    return df


def get_MLE_regression_div_results(results, n_tips, scenario, error):
    reg_values = {
        'BD': ['a', 'r'],
        'HE': ['a', 'r'],
        'ME': ['a', 'r', 'frac', 'time'],
        'SR': ['a0', 'a1', 'r0', 'r1', 'time'],
        'WW': ['a0', 'a1', 'r0', 'r1', 'time'],
        'SAT': ['lambda 0'],
    }
    
    values = results[n_tips][scenario][error]
    if error == "MAE":
        df = pd.DataFrame([values], columns=reg_values[scenario], index=['MAE'])
    else: 
        df = pd.DataFrame([values], columns=reg_values[scenario], index=['MRE'])

    return df


def get_clipping_results(results, n_tips, scenario):
    reg_values = {
        'BD': ['a', 'r'],
        'HE': ['a', 'r'],
        'ME': ['a', 'r', 'frac', 'time'],
        'SR': ['a0', 'a1', 'r0', 'r1', 'time'],
        'WW': ['a0', 'a1', 'r0', 'r1', 'time'],
        'SAT': ['lambda 0'],
    }

    clipped_total = results[n_tips][scenario]["clipped_perc"]
    clipped_below = results[n_tips][scenario]["clipped_below"]
    clipped_above = results[n_tips][scenario]["clipped_above"]

    columns = reg_values[scenario]

    df_total = pd.DataFrame([clipped_total], columns=columns, index=["Clipped %"])
    df_below = pd.DataFrame([clipped_below], columns=columns, index=["Below %"])
    df_above = pd.DataFrame([clipped_above], columns=columns, index=["Above %"])

    return df_total, df_below, df_above
