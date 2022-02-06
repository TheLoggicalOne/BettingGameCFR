# DOCUMANTATION AND COMMENTS TO BE COMPLETED
import numpy as np
from collections import deque, defaultdict, namedtuple

# import random as random
# import copy as copy
# import itertools as it
"""Creation of public nodes as numpy 1*4 array, called np_node and functions that work on them(their name ends with _np)

Here we create nodes for Betting Games with different max_number_of_bets. For more information about BettingGames see 
the project READ.me or the article.
Each public node is 1*4 numpy array. 
node[0] is number of total 2*raise+bet+call by OP player. Our interpretation of this number is total number of OP bets.
node[1] is total number of 2*raise+bet+call by IP player. Our interpretation of this number is total number of IP bets.
node[2] is total number of check+fold
node[3] is depth.
These four information completely characterize each node. We define functions that help get other information about 
np_nodes and create tree structure of nodes
"""
# --------------------------------- GENERAL NODE AND ACTION DEFINITIONS AND FUNCTIONS -------------------------------- #

# Constants
FOLD = 'Fold';
CHECK = 'Check';
CALL = 'Call';
BET = 'Bet';
RAISE = 'Raise'
TERMINAL = 'Terminal';
DECISION = 'Decision';
START = 'Start'
# Check or Fold = 0 --- Bet or Call = 1 --- Raise = 2
IP_ACTIONS = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [0, 2, 0, 1]])
OP_ACTIONS = np.array([[0, 0, 1, 1], [1, 0, 0, 1], [2, 0, 0, 1]])
ALL_ACTIONS = np.array([[[0, 0, 1, 1], [1, 0, 0, 1], [2, 0, 0, 1]], [[0, 0, 1, 1], [0, 1, 0, 1], [0, 2, 0, 1]]])
# ALL_ACTIONS[i][j] is:
# j'th action of i'th player for i= 0,1 (OP, IP) and j = 0, 1, 2 (Check or Fold, Bet or Call, Raise)
ACTION_NAME_TO_ID = dict({'Fold': 0, 'Check': 0, 'Call': 1, 'Bet': 1, 'Raise': 2})

START_NODE = np.array([0, 0, 0, 0])

# RBC_COUNTER.dot(node) returns Total number of 2*Raises+Bet+Call (for both players)
RBC_COUNTER = np.array([1, 1, 0, 0])


# return type of node: (FirstAction, Terminal or Decision node, LastAction)
def history_summary_np(node_np):
    rbc = RBC_COUNTER.dot(node_np)
    d = node_np[0] - node_np[1]
    if rbc > 0:
        if node_np[2] == 2:
            return CHECK, TERMINAL, FOLD
        elif node_np[2] == 1:

            if d == 0:
                return CHECK, TERMINAL, CALL
            else:
                if node_np[0] % 2 == 1:
                    return BET, TERMINAL, FOLD
                elif node_np[0] % 2 == 0:
                    if rbc == 1:
                        return CHECK, DECISION, BET
                    elif rbc > 1:
                        return CHECK, DECISION, RAISE
        elif node_np[2] == 0:
            if rbc == 1:
                return BET, DECISION, BET
            elif rbc > 1:
                if d == 0:
                    return BET, TERMINAL, CALL
                else:
                    return BET, DECISION, RAISE
    elif rbc == 0:
        s = np.sum(node_np)
        if s == 0:
            return None, START, None
        elif s == 2:
            return CHECK, DECISION, CHECK
        elif s == 4:
            return CHECK, TERMINAL, CHECK


def pot_size_np(node_np):
    return 3 ** (node_np[0]) + 3 ** (node_np[1])


def is_terminal_np(node_np):
    return history_summary_np(node_np)[1] == TERMINAL


def to_move_np(node_np):
    return node_np[3] % 2


# return all possible actions(from small to big!) in given (node) subject to (MaxNumberOfRaises).
def np_actions_np(node_np, max_number_of_bets):
    rbc = node_np[0] + node_np[1]
    if is_terminal_np(node_np):
        return []
    else:
        if node_np[0] == node_np[1] == 0:
            return ALL_ACTIONS[to_move_np(node_np)][0:2, :]
        else:
            if rbc + 2 < max_number_of_bets:
                return ALL_ACTIONS[to_move_np(node_np)]
            elif rbc + 2 >= max_number_of_bets:
                return ALL_ACTIONS[to_move_np(node_np)][0:2, :]


# return children of given (node) subject to (MaxNumberOfRaises)
def np_children_np(node_np, MaxNumberOfBets):
    C = []
    for action in np_actions_np(node_np, MaxNumberOfBets):
        C.append(node_np + action)
    return C


# return (other is child of node)
def is_child_np(node, other, max_number_of_bets):  # returns: is other child of node
    for action in np_actions_np(node, max_number_of_bets):
        if np.array_equal(node + action, other):
            return True
    return False


# return (other is parent of node)
def is_parent_np(node, other, max_number_of_bets):  # returns: is other parent of node
    for action in np_actions_np(other, max_number_of_bets):
        if np.array_equal(node, other + action):
            return True
    return False


# -------------------------------------------- CREATING RAW TREE OF NODES -------------------------------------------- #


# create DepthFirst tree starting from (root) subject to (MaxNumberOfRaises). Return the result in a list
def create_depth_first_tree(root, max_number_of_bets):
    list_of_tree_nodes = []

    def _create_children_recursive(node):
        list_of_tree_nodes.append(node)
        if is_terminal_np(node):
            return
        else:
            for action in np_actions_np(node, max_number_of_bets):
                _create_children_recursive(node + action)

    _create_children_recursive(root)
    return list_of_tree_nodes


# create BreathFirst tree starting from (root) subject to (MaxNumberOfRaises). Return the result in a list
def create_breadth_first_tree(root, max_number_of_bets):
    list_of_tree_nodes = []
    queue_of_nodes = deque()
    queue_of_nodes.append(root)
    while len(queue_of_nodes) > 0:
        if is_terminal_np(queue_of_nodes[0]):
            list_of_tree_nodes.append(queue_of_nodes.popleft())
        else:
            for action in np_actions_np(queue_of_nodes[0], max_number_of_bets):
                queue_of_nodes.append(queue_of_nodes[0] + action)
            list_of_tree_nodes.append(queue_of_nodes.popleft())
    return list_of_tree_nodes

