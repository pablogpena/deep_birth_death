# -*- coding: utf-8 -*-
# +
import os, pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from time import time
import math

import seaborn as sns
import matplotlib.pyplot as plt


# -

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

def get_regression_metrics(results, n_tips, scenario, error):
    reg_values = {
        'BD': ['r', 'a'],
        'HE': ['r', 'a'],
        'ME': ['r', 'a', 'time', 'frac'],
        'SR': ['r0', 'r1', 'a0', 'a1', 'time'],
        'WW': ['r0', 'r1', 'a0', 'a1', 'time'],
        'SAT': ['lambda 0'],
    }
    
    values = results[n_tips][scenario][error]

    df = pd.DataFrame([values], columns=reg_values[scenario], index=[str(error)])

    return df


# # Errors plots

def plot_errors(results, n_tips, evo_type, save_path=None):
    if evo_type in ("BD", "HE"):
        param_names = ["r", "a"]
        width, height = 8, 5
    elif evo_type == "ME":
        param_names = ["r", "a", "time", "rho"]
        width, height = 20, 5
    elif evo_type == "SAT":
        param_names = ["lambda0"]
        width, height = 3, 5
    else:
        param_names = ["r0", "r1", "a0", "a1", "time"]
        width, height = 20, 5

    errors = results[n_tips][evo_type]["raw_error"]
    num_params = errors.shape[1]

    width = max(4 * num_params, 6)
    height = 5
    fig, axes = plt.subplots(1, num_params, figsize=(width, height), facecolor='white')

    if num_params == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        values = errors[:, i]
        df_plot = pd.DataFrame({
            "Error": values,
            "Parameter": [param_names[i]] * len(values)
        })

        sns.swarmplot(data=df_plot, x="Parameter", y="Error", size=3, color="steelblue", ax=ax)
        ax.set_title(f"{evo_type} — {param_names[i]}")
        ax.set_xlabel("")
        ax.set_ylabel("Error")
        ax.set_xticklabels([])

    plt.subplots_adjust(wspace=0.75)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_errors_boxplot(results, n_tips, evo_type, save_path=None):
    if evo_type in ("BD", "HE"):
        param_names = ["r", "a"]
        width, height = 8, 5
    elif evo_type == "ME":
        param_names = ["r", "a", "time", "rho"]
        width, height = 20, 5
    elif evo_type == "SAT":
        param_names = ["lambda0"]
        width, height = 3, 5
    else:
        param_names = ["r0", "r1", "a0", "a1", "time"]
        width, height = 20, 5

    errors = results[n_tips][evo_type]["raw_error"]
    num_params = errors.shape[1]

    fig, axes = plt.subplots(1, num_params, figsize=(width, height), facecolor='white')

    if num_params == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.boxplot(errors[:, i], showmeans=True)
        ax.set_title(f"{evo_type} — {param_names[i]}")
        ax.set_xticklabels([])

    plt.subplots_adjust(wspace=0.5)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def predicted_minus_target_vs_target(results, tip, evo_type):

    if evo_type == "BD" or evo_type == "HE":
        param_names = ["r", "a"]
    elif evo_type == "ME": 
        param_names = ["r", "a", "time", "rho"]
    elif evo_type == "SAT":
        param_names = ["lambda0"]
    else: 
        param_names = ["r0", "r1", "a0", "a1", "time"]
    
    n_params = len(param_names)
    n_cols = min(n_params, 3)
    n_rows = math.ceil(n_params / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
    
    sns.set_style('white')
    sns.set_context('talk')
    
    if isinstance(axes, plt.Axes):
        axes_list = [axes]
    else:
        axes_list = axes.flatten().tolist()

    errors = results[tip][evo_type]["abs_error"]
    y_test = results[tip][evo_type]['real_values']

    y_test = np.array(y_test)

    for i, ax in enumerate(axes_list):
        if i < n_params:
            param_name = param_names[i]
            
            if errors.ndim == 2:
                err_i = errors[:, i]
            else:
                err_i = errors
            
            sns.regplot(x=y_test[:, i], y=err_i, ci=95, n_boot=500, 
                        scatter_kws={'s': 2, 'color': 'grey'}, 
                        line_kws={'color': 'orange', 'linewidth': 2}, ax=ax)

            innerlimit = min(y_test[:, i])
            outerlimit = max(y_test[:, i])
            ax.plot([innerlimit, outerlimit], [0, 0], linewidth=2, color='purple')

            ax.set_title(param_name)
            ax.set_xlabel('target')
            ax.set_ylabel('target - predicted')
        else:
            ax.axis('off')

    fig.suptitle(f'Target vs (Target-Predicted) for {tip} tips, {evo_type} diversification scenario', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


# # Clipping percentages

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
