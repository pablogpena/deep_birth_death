"""Based on check_env.ipynb"""

import random

from ete3 import Tree
import pandas as pd
import numpy as np

from _encoding import (
    name_tree,
    rescale_branch_lengths,
    add_dist_to_root,
    _increased_recursion_limit,
)


DIVERSIFICATION_SCORE = "diversification_score"
TURN_ONE = "turn_one"


def lambert_encodings(
    tree_file, max_len: int = None, tree_formats: int = 1, file_encoding: str = "utf-8"
):
    with open(tree_file, encoding=file_encoding) as file:
        forest = file.read().replace("\n", "")
    trees = forest.split(";")  # split to individual trees

    # encode tree by tree
    for i, tree_string in enumerate(trees):
        if not tree_string:
            continue

        tree = Tree(tree_string + ";", format=tree_formats)
        name_tree(tree)

        # rescale tree to average branch length of 1
        # measure average branch length
        rescale_factor = get_average_branch_length(tree)
        # rescale tree
        rescale_branch_lengths(tree, rescale_factor)

        # add dist to root attribute
        tr_height = add_dist_to_root(tree)

        # add pathway of visiting priorities for encoding
        add_diversification(tree)
        add_diversification_sign(tree)

        # encode the tree
        with _increased_recursion_limit():
            tree_embedding = list(enc_diver(tree))

        # add tree height
        tree_embedding.insert(0, tr_height)

        # complete embedding
        if max_len is not None:
            tree_embedding = add_padding(tree_embedding, max_len)

        # add type count and scaling factor
        tree_embedding.extend([rescale_factor])

        line_df = pd.DataFrame(tree_embedding, columns=[i])

        if i == 0:
            result = line_df
        else:
            result = pd.concat([result, line_df], axis=1)

    result = result.T

    return result


def get_average_branch_length(tre) -> float:
    """
    Returns average branch length for given tree
    :param tre: ete3.Tree, the tree on which we measure the branch length
    :return: float, average branch length
    """
    br_length = [nod.dist for nod in tre.traverse()]
    return float(np.average(br_length))


def add_diversification(tr) -> None:
    """Adds an attribute, 'diversification_score', i.e. the sum of
    pathways of branched tips to each node.

    Args:
        tr: ete3.Tree, the tree to be modified
    """
    for node in tr.traverse("postorder"):
        if node.is_root():
            continue
        if node.is_leaf():
            setattr(node, DIVERSIFICATION_SCORE, node.dist)
        else:
            children = node.get_children()
            setattr(
                node,
                DIVERSIFICATION_SCORE,
                getattr(children[0], DIVERSIFICATION_SCORE)
                + getattr(children[1], DIVERSIFICATION_SCORE),
            )


def add_diversification_sign(tr) -> None:
    """
    Puts topological signatures based on diversification (i.e. longest path): if the first child of a node has longer
    path of branches leading to it, then it is prioritized for visit.
    :param tr: ete3.Tree, the tree to get the topological description
    :return: void, modifies the original tree
    """
    for node in tr.traverse("levelorder"):
        if node.is_leaf():
            continue
        diver_child0 = getattr(node.children[0], DIVERSIFICATION_SCORE)
        diver_child1 = getattr(node.children[1], DIVERSIFICATION_SCORE)
        if diver_child0 < diver_child1:
            node.add_feature(TURN_ONE, True)
        elif diver_child0 == diver_child1:
            next_sign = random.choice([True, False])
            if next_sign is True:
                node.add_feature(TURN_ONE, True)
        else:
            node.add_feature(TURN_ONE, False)


def enc_diver(anc):
    """Encodes the tree in a depth-first manner."""
    leaf = follow_signs(anc)
    setattr(leaf, "visited", True)
    anc = get_not_visited_anc(leaf)
    if anc is None:
        return
    setattr(anc, "visited", True)
    yield get_dist_to_root(anc)

    yield from enc_diver(anc)


def follow_signs(anc):
    """Follows the signs to the next leaf.

    Args:
        anc: The ancestor node.

    Returns:
        The next leaf.
    """
    end_leaf = anc
    while not end_leaf.is_leaf():
        turn_one = getattr(end_leaf, TURN_ONE, False)
        if turn_one:
            is_node_one_visited = getattr(
                end_leaf.children[1], "visited", False
            )
            if is_node_one_visited:
                end_leaf = end_leaf.children[0]
            else:
                end_leaf = end_leaf.children[1]
        else:
            is_node_zero_visited = getattr(
                end_leaf.children[0], "visited", False
            )
            if is_node_zero_visited:
                end_leaf = end_leaf.children[1]
            else:
                end_leaf = end_leaf.children[0]
    return end_leaf


def get_not_visited_anc(leaf):
    """Returns the first ancestor that has not been visited."""
    while getattr(leaf, "visited", False):
        leaf = leaf.up
    return leaf


def get_dist_to_root(anc):
    """Returns the distance to the root of the tree."""
    dist_to_root = getattr(anc, "dist_to_root")
    return dist_to_root


def add_padding(encoding, max_length):
    """Adds padding to the encoding to reach the max_length.

    Args:
        encoding (list): The encoding to pad.
        max_length (int): The maximum length of the encoding.

    Note:
        Renamed from ``complete_encoding``.
    """
    add_vect = np.repeat(0, max_length - len(encoding))
    encoding.extend(add_vect.tolist())
    return encoding
