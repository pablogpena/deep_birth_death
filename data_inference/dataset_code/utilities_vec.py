# +
from ete3 import Tree
from dataset_code.encoding import *

from tqdm import tqdm

import multiprocessing


# -

def read_tree(newick_tree):
    """
    Tries all nwk formats and returns an ete3 Tree

    :param newick_tree: str, a tree in newick format
    :return: ete3.Tree
    """
    
    tree = None
    for f in (3, 2, 5, 0, 1, 4, 6, 7, 8, 9):
        try:
            tree = Tree(newick_tree, format=f)
            tree.ladderize()
            break
        except:
            continue
    if not tree:
        raise ValueError('Could not read the tree {}. Is it a valid newick?'.format(newick_tree))
    return tree


def load_ind_tree(tr,
                  return_node_names: bool = False,
                  remove_tips: bool = True,
                  rescale_tree: bool = True):
    """
    Loads a tree and returns its vectorization in CDV format

    :param str tr: Path to a valid .nwk file or string with the tree in newick format
    :return: array with CDV encoding
    """
    
    tree = read_tree(tr)
    encoding = encode_to_cdv(tree, return_node_names, remove_tips, rescale_tree)
    
    return encoding


def load_trees_from_array(trees, return_resc_factor=False):
    """
    Loads a set of trees and returns their vectorization in CDV format

    :param list trees: List of trees defined either with their path to a .nwk file
           or string with the tree in newick format
    :return: numpy array with CDV encoding of the set of trees 
    """
    
    encoded_trees = []
    rescale_factors = []
    
    pool = multiprocessing.Pool()
    trees_vec_res = list(tqdm(pool.imap(load_ind_tree, trees), total=len(trees)))
    trees_vec_res = np.array(trees_vec_res)
    
    encoded_trees = trees_vec_res[:, 0]
    rescale_factors = trees_vec_res[:, 1]
        
    encoded_trees = pd.concat(encoded_trees, ignore_index=True)
    print(len(encoded_trees), " trees loaded")

    if return_resc_factor:
        return encoded_trees.to_numpy(), rescale_factors
    
    else:
        return encoded_trees.to_numpy()
