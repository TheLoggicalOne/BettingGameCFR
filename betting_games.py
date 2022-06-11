# ---------------------------------------------- STATE OF THIS CODE -------------------------------------------------- #
# STATUS OF CURRENTS CODES: works perfectly, except possibly uniform_strategy and all_raise_strategy
# their working version is in strategy.py anyway

# FUTURE PLAN: just adding last few changes to documentation

# DOCUMENTATION: very good, just should add few last changes
# -------------------------------------------------------------------------------------------------------------------- #
""" This module can be seen as Betting Game API. Here  we can create Betting Games with different specification
parameters. Betting Game Is a simplified model for one street of betting in poker. For more information about betting
game see the ReadMe.md or the article. For general definition and description of the game see ReadMe-Game Description,
for our specific representation of the game that is also used in this API see ReadMe-Game Representation
"""
import numpy as np
import public_nodes_tree as public_nodes_tree
from collections import namedtuple
from itertools import product, permutations
from utilities import START_NODE


class BettingGame:
    """ Creates a betting game object with following specification parameters (inputs):

    max_number_of_bets: int, Total number of bets allowed. Call count as one bet and each Raise count as 2 bets.(since
        technically a raise is a call and then a bet)

    bet_size: float, showing size of bet in relation to pot. This is fix in game and has default value of 1.

    deck: a dictionary {a_i:x_i for i in range(1, m)} where dictionary keys a_i, for i=1,2, ...,m show different
    possible cards(or hands) in the deck, and each dictionary  value `x_i` shows how many of card `a_i` we have in the
    deck.
    # TODO add second deck to inputs
    deal_from_deck_with_substitution: Boolean


    Application and Usage:
    Most useful attributes:
    self.public_state: list, each public state as namedtuple is one element of this list , in order of their
        creation in breath_first tree. This light and fast representation of public states is actually a reference
        to public_node_tree.public_tree.create_PublicState_list()
        Let S=self.public_state, Then S[i] is public node i representation as namedtuple public_nodes_tree.PublicState()
        Fields of this namedtuple let us access almost every information about the node:
            S[i].node: int, equal to i, public node number or name
            S[i].np_node: numpy.ndarray representation of node i
            S[i].actions: ordered list of actions available in node i
            S[i].children: ordered list of children of node i
            S[i].parent: parent of node i
            S[i].to_move: 0 or 1, showing whose player move is at node i
            S[i].is_terminal: boolean
            S[i].first_played_action: 'Check' or 'Bet', showing first action of in the StartNode
            S[i].last_played_action: 'Check' or 'Call' or 'Bet' or 'Raise', showing The last played action that has led
             us to current node.

    self.node: numpy.ndarray, list of public nodes

    self.next_node: numpy.ndarray, this np array makes moving from one state (public node) to its child easy and fast.
        i.e: self.next_node[i,j] shows the resulting node of taking action j at node i

    some additional attributes:
    self.first_common_ancestors: numpy.ndarray, self.first_common_ancestors[i, j] is a list containing all common nodes
        in the paths from StartNode to current nodes.

    self.history_of_node: list, element i shows action history of node i, showing actions as string

    self.decision_public_state: list, shows all public states that are decision points in order

    To access more information about public tree and nodes use this:
    self.public_tree : public_nodes_tree.PublicTree object, gives the public tree of the game. In case you still need
        more info about public tree and public states this let you access everything about public states and their tree
        structure through methods and attrs available in public_nodes_tree.PublicTree class.
        For more info see public_nodes_tree.PublicTree documentation.

    Following attributes are just creating namedtuple types to be instantiated by corresponding methods:
    self.InfoNode: create the raw namedtuple with fields node and hand, to create namedtuple representation of
        InfoNodes of game. To populate this and actually create info nodes use method: indo_node
    self.InfoState: create the raw namedtuple with fields public_states and hand, to create namedtuple representation of
        InfoStates of game. To populate this and actually create info nodes use method: indo_state
    self.WorldNode: create the raw namedtuple with fields node and op_hand and ip_hand to create namedtuple
        representation of WorldNodes of game. To populate this and actually create info nodes use method: world_node
    self.WorldState: create the raw namedtuple with fields public_states and op_hand and ip_hand to create namedtuple
        representation of WorldStates of game. To populate this and actually create info nodes use method: world_state
        """

    def __init__(self, max_number_of_bets,
                 deck=None,
                 bet_size=1,
                 deal_from_deck_with_substitution=True):
        if deck is None:
            deck = {i: 1 for i in range(100)}
        self.max_number_of_bets = max_number_of_bets
        self.bet_size = bet_size
        self.deck = deck
        self.deal_from_deck_with_substitution = deal_from_deck_with_substitution

        self.number_of_hands = np.array([value for key, value in self.deck.items()]).sum()
        self.name = 'B' + '_' + str(self.max_number_of_bets) + '_' + str(self.number_of_hands) + '_' + str(
            int(self.deal_from_deck_with_substitution)) + '_' + str(
            int(self.bet_size * 100))

# ----------------------------------- Public: Tree, States, Nodes, play, history ------------------------------------- #

        self.public_tree = public_nodes_tree.PublicTree(START_NODE, self.max_number_of_bets)
        self.node = self.public_tree.nodes
        self.public_state = self.public_tree.create_PublicState_list()
        self.depth_of_node = np.array([len(self.path_of_node(i)) - int(i == 0) for i in self.node])
        self.depth_of_tree = self.depth_of_node[-1]
        self.parent = [self.public_tree.parent(i) for i in self.node]
        self.history_of_node = [self.public_tree.history_of_node(i) for i in self.node]
        self.first_common_ancestors = np.array([[self.public_tree.common_ancestors(i, j)[-1] for i in self.node]
                                                for j in self.node])
        self.next_node = self.public_tree.create_next_node_table()

        self.decision_public_state = [self.public_state[i] for i in self.public_tree.decision_nodes]
        self.decision_node = np.array([self.node[i] for i in self.public_tree.decision_nodes])
        self.terminal_public_state = [self.public_state[i] for i in self.node if self.public_state[i].is_terminal]
        self.terminal_node = np.array([self.node[i] for i in self.node if self.public_state[i].is_terminal])

        self.terminal_values_node = np.array([[[
            self.terminal_value(node, op_hand, ip_hand) * (
                        self.deal_from_deck_with_substitution or (op_hand != ip_hand))
            for ip_hand in self.deck.keys()]
            for op_hand in self.deck.keys()]
            for node in self.terminal_node])
        self.terminal_values_all_nodes = np.array([[[
            self.terminal_value(node, op_hand, ip_hand) * (
                        self.deal_from_deck_with_substitution or (op_hand != ip_hand))
            for node in self.node]
            for ip_hand in self.deck.keys()]
            for op_hand in self.deck.keys()], dtype=np.float64)

# -------------------------- Base namedtuple for Information and World States and Nodes ----------------------------- #

        self.InfoNode = namedtuple('InfoNode' + str(self.max_number_of_bets), ['node', 'hand'])
        self.InfoState = namedtuple('InfoState' + str(self.max_number_of_bets), ['public_state', 'hand'])
        self.WorldNode = namedtuple('WorldNode' + str(self.max_number_of_bets), ['node', 'op_hand', 'ip_hand'])
        self.WorldState = namedtuple('State' + str(self.max_number_of_bets),
                                     ['public_state', 'op_hand', 'ip_hand'])

# ----------------------------------- STRATEGY AND VALUES INDUCED BY STRATEGY ---------------------------------------- #

    def uniform_strategy(self):
        S = np.zeros((len(self.deck.keys()), self.public_tree.number_of_nodes))
        for i in range(len(self.deck.keys())):
            for _decision_node in self.public_tree.decision_nodes:
                for child in self.public_state[_decision_node].children:
                    S[i, child] = 1 / len(self.public_state[_decision_node].children)
        return S

    def all_raise_strategy(self):
        S = np.zeros((len(self.deck.keys()), self.public_tree.number_of_nodes))
        for i in range(len(self.deck.keys())):
            for _decision_node in self.public_tree.decision_nodes:
                for child in self.public_state[_decision_node].children:
                    S[i, child] = int(child == max(self.public_state[_decision_node].children))
        return S

# ---------------------------------------------------OTHER METHODS---------------------------------------------------- #

    def path_of_node(self, node):
        path = [i for i in self.public_tree.ancestors_of_node(node) if i > 0]
        path.append(node)
        return path

    def terminal_value(self, node, op_hand, ip_hand):
        if not self.public_state[node].is_terminal:
            return 0
        pot_multiplier = 1 + 2 * self.bet_size
        np_node = self.public_tree.np_rep_of_nodes[node]
        if np_node[0] > np_node[1]:
            return pot_multiplier ** (np_node[1])
        elif np_node[0] < np_node[1]:
            return -pot_multiplier ** (np_node[0])
        elif np_node[0] == np_node[1]:
            if op_hand > ip_hand:
                return pot_multiplier ** (np_node[0])
            elif op_hand < ip_hand:
                return -pot_multiplier ** (np_node[0])
            else:
                return 0

    def cards_dealing(self):
        number_of_cards = len(self.deck.keys())
        cards = []
        deck_of_paired_cards = []
        for key, value in self.deck.items():
            for i in range(value):
                cards.append(key)
        if self.deal_from_deck_with_substitution:
            deck_of_paired_cards = list(product(cards, cards))
        elif not self.deal_from_deck_with_substitution:
            for index_one, index_two in permutations(range(len(cards)), 2):
                deck_of_paired_cards.append((cards[index_one], cards[index_two]))

        return deck_of_paired_cards

    def deck_matrix(self):
        n_cards = len(self.deck.keys())
        matrix = np.ones((n_cards, n_cards))
        for i in range(n_cards):
            for j in range(n_cards):
                matrix[i, j] = self.deck[i]-((1-int(self.deal_from_deck_with_substitution))*(i == j))
        return matrix/np.sum(matrix)

# implementation of deck_matrix using np.fromfunction did not work for some reason
#   def deck_matrix(self):
#       n_cards = len(self.deck.keys())
#       M = np.fromfunction(
#           lambda i, j: self.deck[i]-((1-int(self.deal_from_deck_with_substitution))*(i == j)), (n_cards, n_cards),
#           dtype=float)
#       return M/(np.sum(M))

# ---------------------------------------Information and World States and Nodes -------------------------------------- #

    def info_node(self, hand):
        """ Creates a list, i'th element of list is self.InfoNode(node=i, hand=hand), namedtuple representation of info
            node corresponding to public node i holding the given hand. """

        return [self.InfoNode(node=node, hand=hand) for node in self.node]

    def info_state(self, hand):
        """ Creates a list, i'th element of list is self.InfoState(public_state=self.public_state[i], hand=hand),
            namedtuple representation of info
            node corresponding to public node i holding the given hand """
        return [self.InfoState(public_state=self.public_state[node], hand=hand)
                for node in self.node]

    def world_node(self, op_hand, ip_hand):
        """ Creates a list, i'th element of list is self.WorldNode(node=node, op_hand=op_hand, ip_hand=ip_hand),
            namedtuple representation of world node corresponding to public node i, players holding the given hands """
        return [self.WorldNode(node=node, op_hand=op_hand, ip_hand=ip_hand) for node in self.node]

    def world_state(self, op_hand, ip_hand):
        """ Creates a list, i'th element of list is
            self.WorldState(public_state=self.public_state[i], op_hand=op_hand, ip_hand=ip_hand),
            namedtuple representation of world state corresponding to public state i holding the given hands """
        return [self.WorldState(public_state=self.public_state[node], op_hand=op_hand, ip_hand=ip_hand)
                for node in self.node]


# ------------------------------------ INITIALIZING BETTING GAMES WITH USUAL SIZES ----------------------------------- #

# Start a Game
if __name__ == '__main__':
    # Creating Kuhn Poker
    J = 0
    Q = 1
    K = 2
    KUHN_BETTING_GAME = BettingGame(bet_size=0.5, max_number_of_bets=2,
                                    deck={J: 1, Q: 1, K: 1}, deal_from_deck_with_substitution=False)
    K = KUHN_BETTING_GAME
    # Creating betting games with other sizes
    max_n = 12
    G = BettingGame(max_n)
    dm = K.deck_matrix()


