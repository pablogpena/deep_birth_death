# +
# #!/usr/bin/env python3

import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import ete3

# all branches of given tree will be rescaled to TARGET_AVG_BL
TARGET_AVG_BL = 1


@contextmanager
def _increased_recursion_limit(limit: int = 100000):
    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(limit)
        yield
    finally:
        sys.setrecursionlimit(original_limit)


def add_dist_to_root(tre: ete3.Tree):
    """Adds a distance to root attribute (dist_to_root) to each node in the
    tree.

    This function traverses the tree in preorder and calculates the distance
    from each node to the root. The distance is stored in a new attribute
    'dist_to_root' for each node.

    Args:
        tre: The tree on which the dist_to_root attribute should
            be added.

    Returns:
        float: The distance to the root of the last leaf node processed,
        which represents the height of the tree.
    """
    tree_height = 0
    for node in tre.traverse("preorder"):
        if node.is_root():
            node.add_feature("dist_to_root", 0)
        elif node.is_leaf():
            node.add_feature(
                "dist_to_root", getattr(node.up, "dist_to_root") + node.dist
            )
            # tips_dist.append(getattr(node.up, "dist_to_root") + node.dist)
            tree_height = getattr(node, "dist_to_root", False)
        else:
            node.add_feature(
                "dist_to_root", getattr(node.up, "dist_to_root") + node.dist
            )
            # int_nodes_dist.append(getattr(node.up, "dist_to_root") +
            # node.dist)
    return tree_height


def name_tree(tre: ete3.Tree, name_tips: bool = True) -> None:
    """Names all the tree nodes that are not named, with unique names.

    Args:
        tre (ete3.Tree): The tree to be named.
        name_tips (bool): Whether to name the tips of the tree.
    """
    i = 0
    for node in tre.traverse("levelorder"):
        if not name_tips and node.is_leaf():
            continue
        # if node.name != "":
        #     continue
        node.name = i
        i += 1


def rescale_branch_lengths(tre: ete3.Tree, target_avg_length: float) -> float:
    """Rescales the branch lengths of a tree to achieve a target average
    branch length.

    Args:
        tre: The tree whose branch lengths are to be rescaled.
        target_avg_length: The target average branch length.

    Returns:
        float: The rescaling factor used to adjust the branch lengths.
    """
    branch_lengths = [node.dist for node in tre.traverse("levelorder")]

    average_branch_length = np.mean(branch_lengths)

    resc_factor = average_branch_length / target_avg_length

    for node in tre.traverse():
        node.dist = node.dist / resc_factor

    return float(resc_factor)


def create_real_polytomies(tre: ete3.Tree) -> None:
    """Replaces internal nodes of zero length with real polytomies.

    Args:
        tre: The tree to be modified.
    """
    for node in tre.traverse("postorder"):
        if node.is_leaf() or node.is_root():
            continue
        if node.dist != 0:
            continue
        for child in node.children:
            node.up.add_child(child)
        node.up.remove_child(node)


def _get_not_visited_anc(leaf: ete3.TreeNode):
    while getattr(leaf, "visited", 0) >= len(leaf.children) - 1:
        leaf = leaf.up
        if leaf is None:
            break
    return leaf


def _get_deepest_not_visited_tip(anc: ete3.TreeNode):
    max_dist = -1
    tip = None
    for leaf in anc:
        if leaf.visited != 0:
            continue
        distance_leaf = _get_dist_to_anc(leaf, anc)
        if distance_leaf > max_dist:
            max_dist = distance_leaf
            tip = leaf
    return tip


def _get_dist_to_root(anc: ete3.TreeNode):
    dist_to_root = getattr(anc, "dist_to_root")
    return dist_to_root


def _get_dist_to_anc(feuille, anc):
    dist_to_anc = getattr(feuille, "dist_to_root") - getattr(
        anc, "dist_to_root"
    )
    return dist_to_anc


def _encode(anc: ete3.TreeNode, yield_names: bool = False):
    leaf = _get_deepest_not_visited_tip(anc)
    dist = _get_dist_to_anc(leaf, anc)
    if not yield_names:
        yield dist
    else:
        yield (dist, leaf.name)
    leaf.visited += 1
    anc = _get_not_visited_anc(leaf)

    if anc is None:
        return
    anc.visited += 1

    if not yield_names:
        yield _get_dist_to_root(anc)
    else:
        yield (_get_dist_to_root(anc), anc.name)

    yield from _encode(anc, yield_names=yield_names)


def encode_to_cdv(
    tree_input: ete3.Tree,
    return_node_names: bool = False,
    remove_tips: bool = True,
    rescale_tree: bool = True,
):
    """Rescales all trees from tree_file so that mean branch length is 1,
    then encodes them into full CDV tree representation.

    Args:
        tree_input: The tree that will be represented in the formof a vector.
        return_node_names: Whether to return the node names in the output.

    Returns:
        pd.DataFrame: Encoded rescaled input trees in the form of most recent,
        with the last column being the rescale factor.
    """

    tree = tree_input.copy()

    # remove the edge above root if there is one
    if len(tree.children) < 2:
        tree = tree.children[0]
        tree.detach()

    create_real_polytomies(tree)

    if rescale_tree:
        rescale_factor = rescale_branch_lengths(
            tree, target_avg_length=TARGET_AVG_BL
        )
    else:
        rescale_factor = 1

    for node in tree.traverse():
        setattr(node, "visited", 0)

    name_tree(tree, name_tips=False)
    tr_height = add_dist_to_root(tree)

    with _increased_recursion_limit():
        if return_node_names:
            tree_embedding, node_names = zip(
                *list(_encode(tree, yield_names=True))
            )
            tree_embedding_list = list(tree_embedding)
            node_names_list: list[str] = list(node_names)
        else:
            tree_embedding_list = list(_encode(tree))

    tree_embedding_list.insert(0, tr_height)

    result = pd.DataFrame(tree_embedding_list, columns=[0])
    result = result.T

    if "1501" in result.columns:
        result = result.drop(columns=["1501", "1502"])
    elif "401" in result.columns:
        result = result.drop(columns=["401", "402"])

    # Delete odd columns (tips)
    if remove_tips:
        result = result.iloc[:, ::2]

    result = result.rename(columns={0: "tree_height"})
    if return_node_names:
        # Rename the other columns to node names
        result = result.rename(
            columns=dict(enumerate(node_names_list, start=1))
        )

    return result, rescale_factor
