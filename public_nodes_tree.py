# DOCUMENTATION AND COMMENTS TO BE COMPLETED
import copy
import numpy as np
import numpy_representation_of_nodes as numpy_representation_of_nodes
from collections import defaultdict, namedtuple
from utilities import START_NODE

# ------------------------------------------ CREATING FINAL TREE AND GAMES ------------------------------------------- #


# a Tree(StartNode, n) will have 3*n+3 nodes. Nodes with ID 0,1 and 2,4 and 3k+1 are non terminal except 3n+1 which will
# exceed MaxNumberOfRaises
# 3n+1 for n>=1 is decision node. Its child are 3n+1+4, 3n+1+5, 3n+1+6.
# Even n:  its bet   type. n = 2*R
# Odd  n:  its check type. n = 2*R+B
class PublicTree:
    """ This is the tree of public nodes. Each node is named by its order of creation in breadth first tree """

    def __init__(self, root, max_number_of_bets):
        self.root = root
        self.max_number_of_bets = max_number_of_bets

        # numpy.ndarray: List containing nodes of Tree in BF order
        self.np_rep_of_nodes = np.array(
            numpy_representation_of_nodes.create_breadth_first_tree(self.root, self.max_number_of_bets))

        # numpy.ndarray: List containing nodes of Tree in DF order
        self.depth_first_np_nodes = np.array(
            numpy_representation_of_nodes.create_depth_first_tree(self.root, self.max_number_of_bets))

        # number of nodes
        self.number_of_nodes = len(self.np_rep_of_nodes)

        # numpy.ndarray: List of IDs of Nodes: 0 to 3n+2
        self.nodes = np.array(range(self.number_of_nodes))

        # a List that x'th element shows BF ID of x'th node in DF Tree. i.e:
        # x'th element of DF tree is what element in BF tree
        self.node_of_depth_first_indexes = np.array([self.node_of_depth_first_index(x)
                                                     for x in range(self.number_of_nodes)])

        # numpy.ndarray: List of all IDs ( BF IDs) that are non-terminal and there is a decision to be made
        self.decision_nodes = np.array([i for i in self.nodes
                                        if not numpy_representation_of_nodes.is_terminal_np(self.np_rep_of_nodes[i])])

        # Number of non-terminal nodes ( decision nodes)
        self.number_of_decision_nodes = len(self.decision_nodes)

        # a List that i'th element shows all the actions of node with ID = i ( BF ID). actions are 1*4 np.array
        self.np_actions_of_nodes = [
            numpy_representation_of_nodes.np_actions_np(self.np_rep_of_nodes[i], self.max_number_of_bets)
            for i in self.nodes]

        # a List that i'th element shows all the action numbers (0,1,2 in order) of node with ID = i ( BF ID).
        self.actions_of_nodes = [list(range(len(self.np_actions_of_nodes[i]))) for i in self.nodes]

        # Dict: keys are node IDs, key=i corresponds to a  value that itself is a dict which contains info of i'th node
        # self.IDInfo[i] is a Dict with keys: 'Node', 'Type', 'Turn', 'Action', 'ActionIDs, 'IsTerminal', 'ChildrenID'
        self.nodes_info = {i: {'NpNode': self.np_rep_of_nodes[i],
                               'HistorySummary': numpy_representation_of_nodes.history_summary_np(
                                   self.np_rep_of_nodes[i]),
                               'ToMove': numpy_representation_of_nodes.to_move_np(self.np_rep_of_nodes[i]),
                               'NpActions': self.np_actions_of_nodes[i], 'Actions': self.actions_of_nodes[i],
                               'IsTerminal': numpy_representation_of_nodes.is_terminal_np(self.np_rep_of_nodes[i]),
                               'Children': [self.node_of_np_node(child) for child in
                                            numpy_representation_of_nodes.np_children_np(self.np_rep_of_nodes[i],
                                                                                         self.max_number_of_bets)]
                               }
                           for i in self.nodes}

        # List that its i'th element shows children of node i
        self.children_of_nodes = [self.nodes_info[i]['Children'] for i in self.nodes]

        # self.IDPlay[(ID , ai)] shows the node ID of playing action (ai) in node with ID=ID
        # gives 0 if (ID, ai) is not a legal (node ID , action number)
        self.next_node_dict = defaultdict(int, {(ID, ai): self.nodes_info[ID]['Children'][ai] for ID in self.nodes
                                                for ai in self.actions_of_nodes[ID]})

        self.next_node_table = self.create_next_node_table()

        # 2D list that [i,j] element is CommonHistory of nodes with IDs: i , j
        self.common_ancestors_table = [[self.common_ancestors(i, j) for j in self.nodes] for i in self.nodes]

        # define namedtuple with typename = 'Pstate' where n is self.MaxNumberOfRaises
        # This namedtuple is stored in self.NamedTuple and its field are 'node actions turn is_terminal children'
        # You can create instance of this for the given tree by calling func: CreateNamedTuple(tree)
        self.PublicState = namedtuple('PublicState' + str(self.max_number_of_bets),
                                      ['node', 'np_node', 'actions', 'children', 'parent', 'to_move',
                                       'is_terminal', 'first_played_action', 'last_played_action'])

    # returns of node given np_node
    def node_of_np_node(self, node):
        for i in self.nodes:
            if np.array_equal(self.np_rep_of_nodes[i], node):
                return i

    # returns BF ID of x'th node in DF Tree. i.e: x'th element of DF tree is what element in BF tree
    def node_of_depth_first_index(self, x):
        return self.node_of_np_node(self.depth_first_np_nodes[x])

    # returns id of Parent of node with id = ID
    def parent(self, node):
        for possible_parent in self.nodes:
            if node in self.nodes_info[possible_parent]['Children']:
                return possible_parent

    # returns index of ID in self.DecisionIDs
    def index_in_decision_nodes(self, node):
        for index in self.decision_nodes:
            if self.decision_nodes[index] == node:
                return index
        else:
            print(f'{node} is not for a Decision Node')

    # return a list that shows all the node ids from StartNode to given ID
    def ancestors_of_node(self, node):
        backward_ancestors_list = []
        while node > 0:
            node = self.parent(node)
            backward_ancestors_list.append(node)
        backward_ancestors_list.reverse()
        ancestors_list = backward_ancestors_list
        return ancestors_list

    # return common history (node ids) of nodes with ids: ID1 , ID2
    def common_ancestors(self, node_a, node_b):

        a = copy.copy(self.ancestors_of_node(node_a))
        b = copy.copy(self.ancestors_of_node(node_b))
        a.append(node_a)
        b.append(node_b)
        c = []
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                break
            c.append(a[i])
        else:
            if len(a) == len(b) and len(a) > 1:
                c.pop()

        return c

    def history_of_node(self, node):
        node_history = copy.deepcopy(self.ancestors_of_node(node))
        node_history.append(node)
        node_history_after_start_node = [node_history[i] for i in range(1, len(node_history))]
        return [self.nodes_info[node]['HistorySummary'][2] for node in node_history_after_start_node]

    # Returns a list containing tree.NamedTuple representation of tree.Nodes. Fields are:
    # 'node'=node, 'actions'=action IDs, 'turn'=Turn(node), 'is_terminal'=is the node terminal, 'children'=children IDs
    def create_PublicState_list(self):
        S = list(range(self.number_of_nodes))
        for i in range(self.number_of_nodes):
            S[i] = self.PublicState(node=i, np_node=self.np_rep_of_nodes[i],
                                    actions=self.actions_of_nodes[i],
                                    children=self.nodes_info[i]['Children'], parent=self.parent(i),
                                    to_move=numpy_representation_of_nodes.to_move_np(self.np_rep_of_nodes[i]),
                                    is_terminal=(numpy_representation_of_nodes.history_summary_np(
                                        self.np_rep_of_nodes[i])[1] == 'Terminal'),
                                    first_played_action=numpy_representation_of_nodes.history_summary_np(
                                        self.np_rep_of_nodes[i])[0],
                                    last_played_action=numpy_representation_of_nodes.history_summary_np(
                                        self.np_rep_of_nodes[i])[2])
        return S

    # np.array table, T[i,j] is resulting node ID of playing action j at node with ID=i
    def create_next_node_table(self):
        play_table = np.full((self.number_of_nodes, 3), 0)
        for i in self.nodes:
            for action in self.actions_of_nodes[i]:
                play_table[i, action] = self.nodes_info[i]['Children'][action]
        return play_table


if __name__ == '__main__':
    T2 = PublicTree(START_NODE, 2)
    NT2 = T2.create_PublicState_list()
    T10 = PublicTree(START_NODE, 10)
