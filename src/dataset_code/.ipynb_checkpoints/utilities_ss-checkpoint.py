# +
# #!/usr/bin/env python3

import sys
import argparse

import numpy as np
import pandas as pd
from scipy.stats import linregress

from ete3 import Tree
from collections import Counter
from math import floor

import pathlib

from tqdm import tqdm

# for parallel processing
import multiprocessing

DISTANCE_TO_ROOT = "dist_to_root"

DEPTH = "depth"

LADDER = "ladder"

target_avg_BL = 1

col = []

col_EmmaBranchLengths = [
    'stem_age',  # max height and min height of the tree DONE
    'a_bl_mean', 'a_bl_median', 'a_bl_var',  # mean, median, var length of all branches DONE
    'e_bl_mean', 'e_bl_median', 'e_bl_var',  # mean, median, var length of external branches DONE
    'i_bl_mean_1', 'i_bl_median_1', 'i_bl_var_1',  # piecewise mean/med/var length of internal branches 1st/3 of tree DONE
    'i_bl_mean_2', 'i_bl_median_2', 'i_bl_var_2',  # piecewise mean/med/var length of internal branches 2nd/3 of tree DONE
    'i_bl_mean_3', 'i_bl_median_3', 'i_bl_var_3',  # piecewise mean/med/var length of internal branches 3rd/3 of tree DONE
    'ie_bl_mean_1', 'ie_bl_median_1', 'ie_bl_var_1',  # ratio of e_BL_mean/... and internal branches 1st/3 of tree DONE
    'ie_bl_mean_2', 'ie_bl_median_2', 'ie_bl_var_2',  # ratio of e_BL_mean/... and internal branches 2nd/3 of tree DONE
    'ie_bl_mean_3', 'ie_bl_median_3', 'ie_bl_var_3'  # ratio of e_BL_mean and internal branches 3rd/3 of tree DONE
    ]
col += col_EmmaBranchLengths

col_EmmaTreeTopology = [
    'colless', 'sackin',  # colless, sackin score: DONE
    'wd_ratio', 'delta_w', 'max_ladder',  # mean, median, var length of all branches DONE
    'il_nodes', 'staircaseness_1', 'staircaseness_2',  # mean, median, var length of external branches, DONE
    ]

col += col_EmmaTreeTopology

col_EmmaLTT = [
    'slope', 'slope_1', 'slope_2', 'slope_3', 'slope_1_2', 'slope_2_3',  # slopes and slope ratios
    'mean_b_time_1', 'mean_b_time_2', 'mean_b_time_3' # mean branching times
]
col += col_EmmaLTT

col_EmmaLTT_COOR = [
    'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10',
    'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20',
    'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8', 'y_9', 'y_10',
    'y_11', 'y_12', 'y_13', 'y_14', 'y_15', 'y_16', 'y_17', 'y_18', 'y_19', 'y_20'
]

col += col_EmmaLTT_COOR

col_chains = [
    'number_sumchain', 'mean_sumchain', 'min_sumchain', '1st_decile_sumchain', '2nd_decile_sumchain',
    '3rd_decile_sumchain', '4th_decile_sumchain', 'median_sumchain', '6th_decile_sumchain', '7th_decile_sumchain',
    '8th_decile_sumchain', '9th_decile_sumchain', 'max_sumchain', 'var_sumchain'
]

col += col_chains

col_NB_TIPS = [
    'nb_tips'
]

col += col_NB_TIPS

col_rescale = ['rescale_factor']

col += col_rescale


def rescale_tree(tre, target_avg_length):
    """
    Returns branch length metrics (all branches taken into account and external only)
    :param tre: ete3.Tree, tree on which these metrics are computed
    :param target_avg_length: float, the average branch length to which we want to rescale the tree
    :return: float, rescale_factor
    """
    # branch lengths
    dist_all = [node.dist for node in tre.traverse("levelorder")]

    all_bl_mean = np.mean(dist_all)

    rescale_factor = all_bl_mean/target_avg_length

    for node in tre.traverse():
        node.dist = node.dist/rescale_factor

    return rescale_factor


def name_tree(tre):
    """
    Names all the tree nodes that are not named, with unique names.
    :param tre: ete3.Tree, the tree to be named
    :return: void, modifies the original tree
    """
    i = 0
    for node in tre.traverse('levelorder'):
        node.name = i
        i += 1
    return None


def add_depth_and_get_max(tre):
    """
    adds depth to each node.
    :param tre: ete3.Tree, the tree to which depth should be added
    :return: modifies the original tree + maximum depth
    """
    max_dep = 0
    for node in tre.traverse('levelorder'):
        if not node.is_root():
            if node.up.is_root():
                node.add_feature("depth", 1)
            else:
                node.add_feature("depth", getattr(node.up, "depth", False)+1)
                if getattr(node, "depth", False) > max_dep:
                    max_dep = getattr(node, "depth", False)
    return max_dep


def add_ladder(tre):
    """
    adds ladder score to each node.
    :param tre: ete3.Tree, the tree to which ladder score should be added
    :return: modifies the original tree
    """
    for node in tre.traverse('levelorder'):
        if not node.is_root():
            if node.up.is_root():
                if not node.is_leaf():
                    if node.children[0].is_leaf() or node.children[1].is_leaf():
                        node.add_feature("ladder", 0)
                    else:
                        node.add_feature("ladder", -1)
                else:
                    node.add_feature("ladder", -1)
            else:
                if not node.is_leaf():
                    if node.children[0].is_leaf() and node.children[1].is_leaf():
                        node.add_feature("ladder", 0)
                    elif node.children[0].is_leaf() or node.children[1].is_leaf():
                        node.add_feature("ladder", getattr(node.up, "ladder", False) + 1)
                    else:
                        node.add_feature("ladder", 0)
                else:
                    node.add_feature("ladder", -1)
        else:
            node.add_feature("ladder", -1)
    return None


def add_dist_to_root(tre):
    """
        Add distance to root (dist_to_root) attribute to each node
        :param tre: ete3.Tree, tree on which the dist_to_root should be added
        :return: void, modifies the original tree
    """

    for node in tre.traverse("preorder"):
        if node.is_root():
            node.add_feature("dist_to_root", 0)
        elif node.is_leaf():
            node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
            # tips_dist.append(getattr(node.up, "dist_to_root") + node.dist)
        else:
            node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
            # int_nodes_dist.append(getattr(node.up, "dist_to_root") + node.dist)
    return None


def tree_height(tre):
    """
    Returns the stem age
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: float, stem age
    """
    for leaf in tre:
        stem_age = tre.get_distance(tre, leaf)
        break
    return stem_age


def branches(tre):
    """
    Returns branch length metrics (all branches taken into account and external only)
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: set of floats, metrics on all branches
    """
    dist_all = []
    dist_ext = []

    for node in tre.traverse("levelorder"):
        dist_all.append(node.dist)
        if node.is_leaf():
            dist_ext.append(node.dist)

    all_bl_mean = np.mean(dist_all)
    all_bl_median = np.median(dist_all)
    all_bl_var = np.nanvar(dist_all)

    ext_bl_mean = np.mean(dist_ext)
    ext_bl_median = np.median(dist_ext)
    ext_bl_var = np.nanvar(dist_ext)

    return all_bl_mean, all_bl_median, all_bl_var, ext_bl_mean, ext_bl_median, ext_bl_var


def piecewise_branches(tre, all_max, e_bl_mean, e_bl_median, e_bl_var):
    """
    Returns piecewise branch length metrics
    :param tre: ete3.Tree, tree on which these metrics are computed
    :param all_max: float, stem age
    :param e_bl_mean: float, mean length of external branches
    :param e_bl_median: float, median length of external branches
    :param e_bl_var: float, variance of length of external branches
    :return: list of 18 floats, summary statistics on piecewise branch length
    """
    dist_all_1 = [node.dist for node in tre.traverse("levelorder") if
                  node.dist_to_root < all_max / 3 and not node.is_leaf()]
    dist_all_2 = [node.dist for node in tre.traverse("levelorder") if
                  all_max / 3 <= node.dist_to_root < 2 * all_max / 3 and not node.is_leaf()]
    dist_all_3 = [node.dist for node in tre.traverse("levelorder") if
                  2 * all_max / 3 <= node.dist_to_root and not node.is_leaf()]

    def i_ie_compute(dist_all_list):
        """
        returns piecewise branch length metrics for given list
        :param dist_all_list: list of internal branch lengths (either 1st, 2nd or 3rd third)
        :return: set of 6 floats, branch length metrics
        """
        if len(dist_all_list) > 0:
            i_bl_mean = np.mean(dist_all_list)
            i_bl_median = np.median(dist_all_list)
            i_bl_var = np.nanvar(dist_all_list)

            ie_bl_mean = np.mean(dist_all_list) / e_bl_mean
            ie_bl_median = np.median(dist_all_list) / e_bl_median
            ie_bl_var = np.nanvar(dist_all_list) / e_bl_var

        else:
            i_bl_mean, i_bl_median, i_bl_var = 0, 0, 0
            ie_bl_mean, ie_bl_median, ie_bl_var = 0, 0, 0

        return i_bl_mean, i_bl_median, i_bl_var, ie_bl_mean, ie_bl_median, ie_bl_var

    output = []
    output.extend(i_ie_compute(dist_all_1))
    output.extend(i_ie_compute(dist_all_2))
    output.extend(i_ie_compute(dist_all_3))

    return output


def colless_func(tre):
    """
    Returns colless metric of given tree
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: float, colless metric
    """
    colless_score = 0
    for node in tre.traverse("levelorder"):
        if not node.is_leaf():
            child1, child2 = node.children
            colless_score += abs(len(child1) - len(child2))
    return colless_score


def sackin_func(tre):
    """
    Returns sackin metric
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: float, sackin score computed on the whole tree (sum of this score on all branches)
    """
    sackin_score = 0
    for node in tre.traverse("levelorder"):
        if node.is_leaf():
            sackin_score += int(getattr(node, DEPTH, False))
    return sackin_score


def wd_ratio_delta_w(tre, max_dep):
    """
    Returns two metrics of tree width
    :param tre: ete3.Tree, tree on which these metrics are computed
    :param max_dep: float, maximal depth of tre
    :return: set of two floats, ratio and difference of maximum width and depth
    """
    width_count = np.zeros(max_dep+1)
    for node in tre.traverse("levelorder"):
        if not node.is_root():
            width_count[int(getattr(node, DEPTH))] += 1
    max_width = max(width_count)
    delta_w = 0
    for i in range(0, len(width_count)-1):
        if delta_w < abs(width_count[i]-width_count[i-1]):
            delta_w = abs(width_count[i]-width_count[i-1])
    return max_width/max_dep, delta_w


def max_ladder_il_nodes(tre):
    max_ladder_score = 0
    il_nodes = 0
    for node in tre.traverse("preorder"):
        if not node.is_leaf():
            if node.ladder > max_ladder_score:
                max_ladder_score = node.ladder
            if node.ladder > 0:
                il_nodes += 1
    return max_ladder_score/len(tre), il_nodes/(len(tre)-1)


def staircaseness(tre):
    """
    Returns staircaseness metrics
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: set of two floats, metrics
    """
    nb_imbalanced_in = 0
    ratio_imbalance = []
    for node in tre.traverse("preorder"):
        if not node.is_leaf():
            if abs(len(node.children[0])-len(node.children[1])) > 0:
                nb_imbalanced_in += 1
            if len(node.children[0]) > len(node.children[1]):
                ratio_imbalance.append(len(node.children[1])/len(node.children[0]))
            else:
                ratio_imbalance.append(len(node.children[0]) / len(node.children[1]))
    return nb_imbalanced_in/(len(tre)-1), np.mean(ratio_imbalance)


def ltt_plot(tre):
    """
    Returns an event (branching) matrix
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: np.matrix, branching events
    """
    events = []

    for node in tre.traverse("levelorder"):
        if not node.is_leaf():
            events.append([node.dist_to_root, 1])

    events = np.asmatrix(events)
    events = np.sort(events.view('i8, i8'), order=['f0'], axis=0).view(np.float)

    events[0, 1] = 2
    for j in np.arange(1, events.shape[0]):
        events[j, 1] = float(events[j - 1, 1]) + float(events[j, 1])

    return events


def ltt_plot_comput(tre):
    """
    Returns LTT plot based metrics
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: set of 9 floats, LTT plot based metrics
    """
    # PART 1: compute list of branching events
    events = []
    for node in tre.traverse():
        if not node.is_leaf():
            events.append(node.dist_to_root)
    events.sort()

    ltt = [_+1 for _ in range(1, len(events)+1)] # +1 dur to initial lineage

    # PART2 slope of the whole ltt plot, slope of thirds of the ltt plot
    slope = linregress(ltt, events)[0]
    slope_1 = linregress(ltt[0:int(np.ceil(len(ltt)/3))], events[0:int(np.ceil(len(ltt)/3))])[0]
    slope_2 = linregress(ltt[int(np.ceil(len(ltt) / 3)):int(np.ceil(2 * len(ltt) / 3))],
                         events[int(np.ceil(len(ltt) / 3)):int(np.ceil(2 * len(ltt) / 3))])[0]
    slope_3 = linregress(ltt[int(np.ceil(2 * len(ltt) / 3)):], events[int(np.ceil(2 * len(ltt) / 3)):])[0]

    slope_ratio_1_2 = slope_1/slope_2
    slope_ratio_2_3 = slope_2/slope_3

    all_max = events[-1]

    # PART3 mean branching times

    # all branching times
    branching_times_1 = [event for event in events if event < all_max/3]
    branching_times_2 = [event for event in events if (all_max/3 < event < 2*all_max/3)]
    branching_times_3 = [event for event in events if 2*all_max/3 < event]

    # differences of consecutive branching times leading to mean branching (1st, 2nd and 3rd
    # part) times
    diff_b_times_1 = [branching_times_1[j + 1] - branching_times_1[j] for j in range(len(branching_times_1)-1)]
    diff_b_times_2 = [branching_times_2[j + 1] - branching_times_2[j] for j in range(len(branching_times_2)-1)]
    diff_b_times_3 = [branching_times_3[j + 1] - branching_times_3[j] for j in range(len(branching_times_3)-1)]

    if len(diff_b_times_1) > 0:
        mean_b_time_1 = np.mean(diff_b_times_1)
    else:
        mean_b_time_1 = 0

    if len(diff_b_times_2) > 0:
        mean_b_time_2 = np.mean(diff_b_times_2)
    else:
        mean_b_time_2 = 0

    if len(diff_b_times_3) > 0:
        mean_b_time_3 = np.mean(diff_b_times_3)
    else:
        mean_b_time_3 = 0

    output = [slope, slope_1, slope_2, slope_3, slope_ratio_1_2, slope_ratio_2_3, mean_b_time_1, mean_b_time_2,
              mean_b_time_3]

    return output


def coordinates_comp(events):
    """
    Returns representation of LTT plot under 20 bins (20 x-axis and 20 y axis coordinates)
    :param events: np.matrix, branching and removal events
    :return: list of 40 floats, y- and x-axis coordinates from LTT plot
    """
    binscor = np.linspace(0, events.shape[0], 21)
    y_axis = []
    x_axis = []
    for i in range(len(binscor)-1):
        y_axis.append(np.average(events[floor(binscor[i]):floor(binscor[i+1]), 0]))
        x_axis.append(np.average(events[floor(binscor[i]):floor(binscor[i+1]), 1]))

    y_axis.extend(x_axis)
    return y_axis


def add_height(tre):
    """
    adds height to each internal node.
    :param tre: ete3.Tree, the tree to which height should be added
    :return: void, modifies the original tree
    """
    for node in tre.traverse('postorder'):
        if node.is_leaf():
            node.add_feature("height", 0)
        else:
            max_child = 0
            for child in node.children:
                if getattr(child, "height", False) > max_child:
                    max_child = getattr(child, "height", False)
            node.add_feature("height", max_child+1)
    return None


def compute_chain(node, order=4):
    """
    Return a list of shortest descending path from given node (i.e. 'transmission chain'), of given order at maximum
    :param node: ete3.node, node on which the descending path will be computed
    :param order: int, order of transmission chain
    :return: list of floats, of maximum length (order)
    """
    chain = []
    contin = True # continue
    while len(chain) < order and contin:
        children_dist = [child.dist for child in node.children]

        chain.append(min(children_dist))
        node = node.children[children_dist.index(min(children_dist))]
        if node.is_leaf():
            contin = False
    return chain


def compute_chain_stats(tre, order=4):
    """
    Returns mean, min, deciles and max of all 'transmission chains' of given order
    :param tre: ete3.Tree, tree on which these metrics are computed
    :param order: int, order of transmission chain
    :return: list of floats
    """
    chain_sumlengths = []
    for node in tre.traverse():
        if getattr(node, 'height', False) > (order-1):
            node_chain = compute_chain(node, order=order)
            if len(node_chain) == order:
                chain_sumlengths.append(sum(node_chain))
    sumstats_chain = [len(chain_sumlengths)]
    if len(chain_sumlengths) > 1:
        # mean
        sumstats_chain.append(np.mean(chain_sumlengths))
        # deciles
        sumstats_chain.extend(np.percentile(chain_sumlengths, np.arange(0, 101, 10)))
        # var
        sumstats_chain.append(np.var(chain_sumlengths))
    else:
        sumstats_chain = [0 for i in range(len(col_chains))]
    return sumstats_chain

def load_ind_tree(tr):
    tr = Tree(tr, format=1)
    
    rescale_factor = rescale_tree(tr, target_avg_length=target_avg_BL)
    
    name_tree(tr)
    max_depth = add_depth_and_get_max(tr)
    add_dist_to_root(tr)
    add_ladder(tr)
    
    stem_age = tree_height(tr)

    a_bl_mean, a_bl_median, a_bl_var, e_bl_mean, e_bl_median, e_bl_var = branches(tr)
    i_bl_mean_1, i_bl_median_1, i_bl_var_1, \
    i_bl_mean_2, i_bl_median_2, i_bl_var_2, \
    i_bl_mean_3, i_bl_median_3, i_bl_var_3, \
    ie_bl_mean_1, ie_bl_median_1, ie_bl_var_1, \
    ie_bl_mean_2, ie_bl_median_2, ie_bl_var_2, \
    ie_bl_mean_3, ie_bl_median_3, ie_bl_var_3 = piecewise_branches(tr, stem_age, e_bl_mean, e_bl_median, e_bl_var)
    
    colless = colless_func(tr)
    sackin = sackin_func(tr)
    wd_ratio, delta_w = wd_ratio_delta_w(tr, max_dep=max_depth)
    max_ladder, il_nodes = max_ladder_il_nodes(tr)
    staircaseness_1, staircaseness_2 = staircaseness(tr)
    
    LTT_plot_matrix = ltt_plot(tr)
    
    slope, slope_1, slope_2, slope_3, slope_1_2, slope_2_3, \
    mean_b_time_1, mean_b_time_2, mean_b_time_3 = ltt_plot_comput(tr)
    x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, \
    x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, \
    y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, \
    y_11, y_12, y_13, y_14, y_15, y_16, y_17, y_18, y_19, y_20 = coordinates_comp(LTT_plot_matrix)
    
    nb_tips = len(tr)
    
    add_height(tr)
    
    number_sumchain, mean_sumchain, min_sumchain, first_decile_sumchain, \
    second_decile_sumchain, third_decile_sumchain, fourth_decile_sumchain, \
    median_sumchain, sixth_decile_sumchain, seventh_decile_sumchain, \
    eighth_decile_sumchain, ninth_decile_sumchain, max_sumchain, var_sumchain = compute_chain_stats(tr, order=4)
    
    return rescale_factor, stem_age, a_bl_mean, a_bl_median, a_bl_var, \
    e_bl_mean, e_bl_median, e_bl_var, i_bl_mean_1, i_bl_median_1, i_bl_var_1, \
    i_bl_mean_2, i_bl_median_2, i_bl_var_2, i_bl_mean_3, i_bl_median_3, i_bl_var_3, \
    ie_bl_mean_1, ie_bl_median_1, ie_bl_var_1, ie_bl_mean_2, ie_bl_median_2, ie_bl_var_2, \
    ie_bl_mean_3, ie_bl_median_3, ie_bl_var_3, colless, sackin, wd_ratio, delta_w, \
    max_ladder, il_nodes, staircaseness_1, staircaseness_2,  slope, slope_1, slope_2, slope_3, \
    slope_1_2, slope_2_3, mean_b_time_1, mean_b_time_2, mean_b_time_3, \
    x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, \
    x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, \
    y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, \
    y_11, y_12, y_13, y_14, y_15, y_16, y_17, y_18, y_19, y_20, \
    nb_tips, number_sumchain, mean_sumchain, min_sumchain, first_decile_sumchain, second_decile_sumchain, \
    third_decile_sumchain, fourth_decile_sumchain, median_sumchain, sixth_decile_sumchain, \
    seventh_decile_sumchain, eighth_decile_sumchain, ninth_decile_sumchain, max_sumchain, var_sumchain
    

def load_tree_ss(path):
    extension = pathlib.Path(path).suffix
    if extension == '.csv':
        # read csv file with trees
        df = pd.read_csv(path, sep="|")
        trees = df['tree'].to_list()
    elif extension == '.nwk':
        with open(path, 'r') as file:
            trees = file.readlines()
            
    else:
        raise Exception("File format not recognised.")

    # initialize output table
    summaries = pd.DataFrame(columns=col)

    # encode tree by tree
    pool = multiprocessing.Pool()
    trees_ss_res = list(tqdm(pool.imap(load_ind_tree, trees), total=len(trees)))
    trees_ss_res = np.array(trees_ss_res)
    
    summaries['rescale_factor'] = trees_ss_res[:, 0]
    
    # Sumstats based on branch lengths
    summaries['stem_age'] = trees_ss_res[:, 1]
    
    summaries['a_bl_mean'] = trees_ss_res[:, 2]
    summaries['a_bl_median'] = trees_ss_res[:, 3]
    summaries['a_bl_var'] = trees_ss_res[:, 4]
    summaries['e_bl_mean'] = trees_ss_res[:, 5]
    summaries['e_bl_median'] = trees_ss_res[:, 6]
    summaries['e_bl_var'] = trees_ss_res[:, 7]
    summaries['i_bl_mean_1'] = trees_ss_res[:, 8]
    summaries['i_bl_median_1'] = trees_ss_res[:, 9]
    summaries['i_bl_var_1'] = trees_ss_res[:, 10]
    summaries['i_bl_mean_2'] = trees_ss_res[:, 11]
    summaries['i_bl_median_2'] = trees_ss_res[:, 12]
    summaries['i_bl_var_2'] = trees_ss_res[:, 13]
    summaries['i_bl_mean_3'] = trees_ss_res[:, 14]
    summaries['i_bl_median_3'] = trees_ss_res[:, 15]
    summaries['i_bl_var_3'] = trees_ss_res[:, 16]
    summaries['ie_bl_mean_1'] = trees_ss_res[:, 17]
    summaries['ie_bl_median_1'] = trees_ss_res[:, 18]
    summaries['ie_bl_var_1'] = trees_ss_res[:, 19]
    summaries['ie_bl_mean_2'] = trees_ss_res[:, 20]
    summaries['ie_bl_median_2'] = trees_ss_res[:, 21]
    summaries['ie_bl_var_2'] = trees_ss_res[:, 22]
    summaries['ie_bl_mean_3'] = trees_ss_res[:, 23]
    summaries['ie_bl_median_3'] = trees_ss_res[:, 24]
    summaries['ie_bl_var_3'] = trees_ss_res[:, 25]
    
    # Sumstats based on tree topology
    summaries['colless'] = trees_ss_res[:, 26]
    summaries['sackin'] = trees_ss_res[:, 27]
    summaries['wd_ratio'] = trees_ss_res[:, 28]
    summaries['delta_w'] = trees_ss_res[:, 29]
    summaries['max_ladder'] = trees_ss_res[:, 30]
    summaries['il_nodes'] = trees_ss_res[:, 31]
    summaries['staircaseness_1'] = trees_ss_res[:, 32]
    summaries['staircaseness_2'] = trees_ss_res[:, 33]
    
    # Sumstats based on LTT plot
    summaries['slope'] = trees_ss_res[:, 34]
    summaries['slope_1'] = trees_ss_res[:, 35]
    summaries['slope_2'] = trees_ss_res[:, 36]
    summaries['slope_3'] = trees_ss_res[:, 37]
    summaries['slope_1_2'] = trees_ss_res[:, 38]
    summaries['slope_2_3'] = trees_ss_res[:, 39]
    summaries['mean_b_time_1'] = trees_ss_res[:, 40]
    summaries['mean_b_time_2'] = trees_ss_res[:, 41]
    summaries['mean_b_time_3'] = trees_ss_res[:, 42]
    
    # Sumstats COORDINATES
    summaries['x_1'] = trees_ss_res[:, 43]
    summaries['x_2'] = trees_ss_res[:, 44]
    summaries['x_3'] = trees_ss_res[:, 45]
    summaries['x_4'] = trees_ss_res[:, 46]
    summaries['x_5'] = trees_ss_res[:, 47]
    summaries['x_6'] = trees_ss_res[:, 48]
    summaries['x_7'] = trees_ss_res[:, 49]
    summaries['x_8'] = trees_ss_res[:, 50]
    summaries['x_9'] = trees_ss_res[:, 51]
    summaries['x_10'] = trees_ss_res[:, 52]
    summaries['x_11'] = trees_ss_res[:, 53]
    summaries['x_12'] = trees_ss_res[:, 54]
    summaries['x_13'] = trees_ss_res[:, 55]
    summaries['x_14'] = trees_ss_res[:, 56]
    summaries['x_15'] = trees_ss_res[:, 57]
    summaries['x_16'] = trees_ss_res[:, 58]
    summaries['x_17'] = trees_ss_res[:, 59]
    summaries['x_18'] = trees_ss_res[:, 60]
    summaries['x_19'] = trees_ss_res[:, 61]
    summaries['x_20'] = trees_ss_res[:, 62]
    summaries['y_1'] = trees_ss_res[:, 63]
    summaries['y_2'] = trees_ss_res[:, 64]
    summaries['y_3'] = trees_ss_res[:, 65]
    summaries['y_4'] = trees_ss_res[:, 66]
    summaries['y_5'] = trees_ss_res[:, 67]
    summaries['y_6'] = trees_ss_res[:, 68]
    summaries['y_7'] = trees_ss_res[:, 69]
    summaries['y_8'] = trees_ss_res[:, 70]
    summaries['y_9'] = trees_ss_res[:, 71]
    summaries['y_10'] = trees_ss_res[:, 72]
    summaries['y_11'] = trees_ss_res[:, 73]
    summaries['y_12'] = trees_ss_res[:, 74]
    summaries['y_13'] = trees_ss_res[:, 75]
    summaries['y_14'] = trees_ss_res[:, 76]
    summaries['y_15'] = trees_ss_res[:, 77]
    summaries['y_16'] = trees_ss_res[:, 78]
    summaries['y_17'] = trees_ss_res[:, 79]
    summaries['y_18'] = trees_ss_res[:, 80]
    summaries['y_19'] = trees_ss_res[:, 81]
    summaries['y_20'] = trees_ss_res[:, 82]

    summaries['nb_tips'] = trees_ss_res[:, 83]
    print('B', summaries['nb_tips'])
    
    summaries['number_sumchain'] = trees_ss_res[:, 84]
    summaries['mean_sumchain'] = trees_ss_res[:, 85]
    summaries['min_sumchain'] = trees_ss_res[:, 86]
    summaries['1st_decile_sumchain'] = trees_ss_res[:, 87]
    summaries['2nd_decile_sumchain'] = trees_ss_res[:, 88]
    summaries['3rd_decile_sumchain'] = trees_ss_res[:, 89]
    summaries['4th_decile_sumchain'] = trees_ss_res[:, 90]
    summaries['median_sumchain'] = trees_ss_res[:, 91]
    summaries['6th_decile_sumchain'] = trees_ss_res[:, 92]
    summaries['7th_decile_sumchain'] = trees_ss_res[:, 93]
    summaries['8th_decile_sumchain'] = trees_ss_res[:, 94]
    summaries['9th_decile_sumchain'] = trees_ss_res[:, 95]
    summaries['max_sumchain'] = trees_ss_res[:, 96]
    summaries['var_sumchain'] = trees_ss_res[:, 97]

    return summaries
