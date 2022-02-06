# ---------------------------------------------- STATE OF THIS CODE -------------------------------------------------- #
# STATUS OF CURRENTS CODES: works perfectly, tested and validated with examples
#
# FUTURE PLAN: documentation
#
# DOCUMENTATION: zero, need to copy from betting_games and then many editing and adding
#
# LAST UPDATE OF THI BOX: dec 31 22:19 - after completing strategy_for_cfr, family_of_strategy_for_cfr and validating
#                         their results by checking kuhn family of games with MY_FAMILY_OF_BETS
# -------------------------------------------------------------------------------------------------------------------- #

import numpy as np
import public_nodes_tree as public_nodes_tree
from collections import namedtuple
from itertools import product, permutations
from utilities import START_NODE, MY_FAMILY_OF_BETS, STANDARD_FULL_FAMILY_OF_BETS


np.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=300, suppress=None, nanstr=None,

                    infstr=None, formatter=None, sign=None, floatmode=None, legacy=None)


# Just the public tree of the game and its methods and attributes, no card dealing, no bet size
class BettingGamePublicTree:

    def __init__(self, max_number_of_bets):
        self.max_number_of_bets = max_number_of_bets

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
        self.name = "BG_PublicTree" + "_" + "max"+str(self.max_number_of_bets)

    def path_of_node(self, node):
        path = [i for i in self.public_tree.ancestors_of_node(node) if i > 0]
        path.append(node)
        return path


# game tree of the game and its methods and attributes, with card dealing but no bet size
class BettingGameWorldTree(BettingGamePublicTree):

    def __init__(self, max_number_of_bets, deck=None, deal_from_deck_with_substitution=True):
        super().__init__(max_number_of_bets)

        if deck is None:
            deck = {i: 1 for i in range(100)}
        self.deck = deck
        self.deal_from_deck_with_substitution = deal_from_deck_with_substitution

        self.number_of_hands = len(self.deck.keys())

        self.name = "BG_WorldTree" + "_" + "max" + str(self.max_number_of_bets) + "_" + "deck" + str(
            self.number_of_hands) + "_" + "sub" + str(deal_from_deck_with_substitution)

    def deck_matrix(self):
        n_cards = self.number_of_hands
        matrix = np.ones((n_cards, n_cards))
        for i in range(n_cards):
            for j in range(n_cards):
                matrix[i, j] = self.deck[i] - ((1 - int(self.deal_from_deck_with_substitution)) * (i == j))
        return matrix / np.sum(matrix)


# full betting game, game tree and cards dealing and bet size and its methods and attributes
class BettingGame(BettingGameWorldTree):

    def __init__(self, max_number_of_bets, deck=None, deal_from_deck_with_substitution=True, bet_size=1):
        super().__init__(max_number_of_bets, deck, deal_from_deck_with_substitution)

        self.bet_size = bet_size

        self.terminal_values_all_nodes = np.array([[[
            self.terminal_value(node, op_hand, ip_hand) * (
                    self.deal_from_deck_with_substitution or (op_hand != ip_hand))
            for node in self.node]
            for ip_hand in self.deck.keys()]
            for op_hand in self.deck.keys()], dtype=np.float64)
        self.name = "BG" + "_" + "max" + str(self.max_number_of_bets) + "_" + "deck" + str(
            self.number_of_hands) + "_" + "sub" + str(deal_from_deck_with_substitution) + "_" + "bet" + str(
            self.bet_size * 100)

    def terminal_value(self, node, op_hand, ip_hand, _bet_size=None):
        if not self.public_state[node].is_terminal:
            return 0
        if _bet_size is None:
            pot_multiplier = 1 + 2 * self.bet_size
        else:
            pot_multiplier = 1 + 2 * _bet_size
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


class BettingGamesFamily(BettingGameWorldTree):
    def __init__(self, max_number_of_bets, deck=None, deal_from_deck_with_substitution=True,
                 bet_family=STANDARD_FULL_FAMILY_OF_BETS):

        super().__init__(max_number_of_bets, deck, deal_from_deck_with_substitution)
        self.bet_family = bet_family
        self.number_of_bets = len(self.bet_family)
        self.family_of_games = [BettingGame(
            self.max_number_of_bets, self.deck, self.deal_from_deck_with_substitution, bet_size/100)
            for bet_size in self.bet_family]

        self.terminal_values_of_games = np.stack([self.family_of_games[i].terminal_values_all_nodes
                                                   for i in range(self.number_of_bets)])

        self.name = "BG" + "_" + "Family" + "_" + "max" + str(self.max_number_of_bets) + "_" + "deck" + str(
            self.number_of_hands) + "_" + "sub" + str(deal_from_deck_with_substitution) + "_" + "betfamily" + str(
            len(self.bet_family))

    # Seems like has no use case
    # def terminal_value(self, bet_size, node, op_hand, ip_hand):
    #     if not self.public_state[node].is_terminal:
    #         return 0
    #     pot_multiplier = 1 + 2 * bet_size
    #     np_node = self.public_tree.np_rep_of_nodes[node]
    #     if np_node[0] > np_node[1]:
    #         return pot_multiplier ** (np_node[1])
    #     elif np_node[0] < np_node[1]:
    #         return -pot_multiplier ** (np_node[0])
    #     elif np_node[0] == np_node[1]:
    #         if op_hand > ip_hand:
    #             return pot_multiplier ** (np_node[0])
    #         elif op_hand < ip_hand:
    #             return -pot_multiplier ** (np_node[0])
    #         else:
    #             return 0


if __name__ == '__main__':
    bgptk = BettingGamePublicTree(2)
    bgwtk = BettingGameWorldTree(2, {i: 1 for i in range(10)}, False)
    bgk = BettingGame(2, {i:1 for i in range(10)}, False, 1)
    bgfk = BettingGamesFamily(2, {i:1 for i in range(10)}, False, bet_family=STANDARD_FULL_FAMILY_OF_BETS)