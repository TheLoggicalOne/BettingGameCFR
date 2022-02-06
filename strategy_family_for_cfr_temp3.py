# ---------------------------------------------- STATE OF THIS CODE -------------------------------------------------- #
# STATUS OF CURRENTS CODES: works perfectly, tested and validated with examples
#
# FUTURE PLAN: documentation
#
# DOCUMENTATION: zero, need to copy from strategy and then many editing and adding
#
# LAST UPDATE OF THI BOX: dec 31 22:19 - after completing strategy_for_cfr, family_of_strategy_for_cfr and validating
#                         their results by checking kuhn family of games with MY_FAMILY_OF_BETS
# -------------------------------------------------------------------------------------------------------------------- #
import time
import copy
import numpy as np
from tqdm import tqdm
from betting_games_for_cfr import BettingGame, BettingGameWorldTree, BettingGamePublicTree, BettingGamesFamily
from utilities import STANDARD_FULL_FAMILY_OF_BETS, STANDARD_FULL_FAMILY_OF_HANDS


class StrategyFamilyForCfr:

    def __init__(self, game_family, initial_strategy=None):
        self.game = game_family

        self.number_of_hands = self.game.number_of_hands
        self.number_of_nodes = self.game.public_tree.number_of_nodes
        self.bet_family = self.game.bet_family
        self.number_of_bets = len(self.game.bet_family)

        # Game nodes and their basic information
        self.node = self.game.node
        self.decision_node = self.game.decision_node
        self.decision_node_children = [self.game.public_state[i].children for i in self.decision_node]
        self.is_decision_node = [not self.game.public_state[i].is_terminal for i in self.game.node]
        self.check_decision_branch = np.array([node for node in self.decision_node
                                               if self.game.public_state[node].first_played_action == 'Check'])
        self.bet_decision_branch = np.array([node for node in self.decision_node
                                             if self.game.public_state[node].first_played_action == 'Bet'])
        self.start_with_check = [self.game.public_state[node].first_played_action == 'Check' for node in self.game.node]
        self.op_turn_nodes = np.array([node for node in self.game.node if self.game.public_state[node].to_move == 0])
        self.ip_turn_nodes = np.array([node for node in self.game.node if self.game.public_state[node].to_move == 1])
        self.depth_of_node = self.game.depth_of_node
        self.turn = [self.depth_of_node[i] % 2 for i in self.game.node]
        self.reach_player = [1 - (self.depth_of_node[i] % 2) for i in self.game.node]
        self.parent = self.game.parent

        self.terminal_values_of_games = self.game.terminal_values_of_games.copy()

        self.chance_reach_prob = self.game.deck_matrix()

        # This present current given strategy and it is the only attributes that changes
        if initial_strategy is None:
            initial_strategy = self.uniform_strategy_family()
        self.initial_strategy = initial_strategy.copy()
        self.strategy_base = self.initial_strategy.copy()
        self.iteration = 0
        self.cumulative_strategy = np.zeros((self.number_of_bets, 2, self.number_of_hands, self.number_of_nodes))
        self.cumulative_regret = np.zeros((self.number_of_bets, 2, self.number_of_hands, self.number_of_nodes))

        # self.numpy_file_str = 'S' + '_' + self.game.name + '_'
        # np.savez('S' + '_' + self.game.name, self.initial_strategy, self.cumulative_regret, self.cumulative_strategy)

    # ---------------------------- MAIN METHODS: REACH PROBABILITIES OF GIVEN STRATEGY  ---------------------------------- #

    # 9used
    # even cols are none-one op cols
    def check_branch_strategy_family(self, position):
        """ return columns corresponding to decision nodes of check branch of game, in strategy matrix of given player

        First column of each of two table corresponds to column 1 in strategy_base
        even indexed columns are none-one op cols """
        return np.concatenate((self.strategy_base[:, position, :, 1:2],
                               self.strategy_base[:, position, :, 4:self.check_decision_branch[-1] + 1:6]), axis=-1)

    # 9used
    # odd cols are none-one op cols
    def bet_branch_strategy_family(self, position):
        """ return columns corresponding to decision nodes of bet branch of game, in strategy matrix of given player

        First column of each of two table corresponds to  column 2 in strategy_base
        even indexed columns are none-one op cols """
        return np.concatenate((self.strategy_base[:, position, :, 2:3],
                               self.strategy_base[:, position, :, 7:self.bet_decision_branch[-1] + 1:6]), axis=-1)

    # 8used
    def player_reach_probs_of_check_decision_branch_info_nodes_family(self, position):
        """ return reach probability columns of decision nodes in check branch of game for given player

        First column corresponds of each player table to to column 1 in strategy_base  """
        return np.cumprod(self.check_branch_strategy_family(position), axis=-1)

    # 8used
    def player_reach_probs_of_bet_decision_branch_info_nodes_family(self, position):
        """ return reach probability columns of decision nodes in bet branch of game for given player

               First column corresponds of each player table to to column 2 in strategy_base  """
        return np.cumprod(self.bet_branch_strategy_family(position), axis=-1)

    # 7used
    # This is the main calculator of  player reach probs of given info node, vectorized over hands of player
    def player_reach_probs_of_info_node_family(self, node, position):
        """ returns (self.number_of_hands)*1 numpy array where row i corresponds to info node(hand=i, node)"""
        if node == 0:
            PRN = np.ones((self.number_of_bets, self.number_of_hands, 1))
        elif self.start_with_check[node]:
            if self.is_decision_node[node]:
                return self.player_reach_probs_of_check_decision_branch_info_nodes_family(position)[:, :,
                       self.depth_of_node[node] - 1:self.depth_of_node[node]]
            else:
                parent = self.parent[node]
                PRN = self.strategy_base[:, position, :,
                      node:node + 1] * self.player_reach_probs_of_check_decision_branch_info_nodes_family(
                    position)[:, :, self.depth_of_node[parent] - 1:self.depth_of_node[parent]]

        else:
            if self.is_decision_node[node]:
                PRN = self.player_reach_probs_of_bet_decision_branch_info_nodes_family(position)[:, :,
                      self.depth_of_node[node] - 1:self.depth_of_node[node]]
            else:
                parent = self.parent[node]
                PRN = self.strategy_base[:, position, :, node:node + 1] \
                      * self.player_reach_probs_of_bet_decision_branch_info_nodes_family(position)[:, :,
                        self.depth_of_node[parent] - 1:self.depth_of_node[parent]]
        return PRN

    # 6used
    # This just tabularize player_reach_probs_of_info_node method
    # makes 2 table one table for each position, each table has one column( size of number of hands) for each node
    def players_reach_probs_of_info_nodes_table_with_update_family(self):
        """ returns 2*(self.number_of_hands)*(self.number_of_nodes) numpy array """
        PR = np.ones((self.number_of_bets, 2, self.number_of_hands, self.number_of_nodes))
        for i in range(2):
            for node in self.node[1:]:
                PR[:, i, :, node:node + 1] = self.player_reach_probs_of_info_node_family(node, i)
        # This part update self.cumulative_strategy
        self.cumulative_strategy += PR
        return PR

    # TODO: each time update_cumulative_regrets is called, this will be called 6 level deep seems like best place
    #  to update strategy sum is to do it inside this method!

    def players_reach_probs_of_info_nodes_table_family(self):
        """ returns 2*(self.number_of_hands)*(self.number_of_nodes) numpy array """
        PR = np.ones((self.number_of_bets, 2, self.number_of_hands, self.number_of_nodes))
        for i in range(2):
            for node in self.node[1:]:
                PR[:, i, :, node:node + 1] = self.player_reach_probs_of_info_node_family(node, i)
        return PR

    # 4used
    # Create 3D table,o for given player, containing cf reach probs( opponent reach probs) of world nodes
    # world node reach probs for each player are equal to info node reach probs
    # creation is by broadcasting player_reach_probs_of_info_nodes_table for one player, by repeating it for each
    # opponent possible hand
    def cf_reach_probs_of_world_nodes_table_family(self, position):
        """ returns (self.number_of_hands)*(self.number_of_hands)*(self.number_of_nodes) numpy array """

        if position == 0:
            reach_of_info_nodes_table = self.players_reach_probs_of_info_nodes_table_with_update_family()[:, 1:2, :, :]
            OR = np.broadcast_to(reach_of_info_nodes_table,
                                 (
                                     self.number_of_bets, self.number_of_hands, self.number_of_hands,
                                     self.number_of_nodes))
        else:
            reach_of_info_nodes_table = self.players_reach_probs_of_info_nodes_table_with_update_family()[:, 0:1, :,
                                        :].reshape(
                self.number_of_bets, self.number_of_hands, 1, self.number_of_nodes)
            OR = np.broadcast_to(reach_of_info_nodes_table,
                                 (
                                     self.number_of_bets, self.number_of_hands, self.number_of_hands,
                                     self.number_of_nodes))
        return OR

    # TODO: 4 - possible speed improvement - analyze possibility of better vectorization

    # TODO: 5 - possible speed improvement - analyze possibility of vectorization

    # ---------------------------------- MAIN METHODS: EVALUATION OF GIVEN STRATEGY -------------------------------------- #
    """   Note One: all cf values and regrets can be computed by values_of_world_nodes_table 
                    and cf_reach_probs_of_world_nodes_table(position)

          Note Two: whenever v is cf_value_world_nodes_table or cf_regrets_of_public_node_from_world_values or ...

                    position info node values can be computed by np.sum(v[:,:,node], axis=1-position)
                    op     info node values can be computed by np.sum(v[:,:,node], axis=1         )
                    ip     info node values can be computed by np.sum(v[:,:,node], axis=0         )
                    also if you drop the node will give a number_of_hands*number_of_nodes table where
                    each column is info node values of corresponding node=column_index
                    each row corresponds to hand in info node      
    """

    # 4used
    def values_of_world_nodes_table_family(self):
        """ returns (self.number_of_hands)*(self.number_of_hands)*(self.number_of_nodes) numpy array """

        number_of_hands = self.number_of_hands
        world_state_values = self.terminal_values_of_games.copy()
        given_strategy = self.strategy_base[:, 0, :, :] * self.strategy_base[
                                                          :, 1, :, :]
        # world_state_value[ , , t.parent] += world_state_value[ , , t]**strategy[ , t]
        nonzero_nodes = self.node[1:]
        reverse_node_list = nonzero_nodes[::-1]

        for current_node in reverse_node_list:
            current_player = self.turn[current_node]
            parent_node = self.parent[current_node]
            parent_node_player = 1 - current_player

            # if parent_node player is op we multiply rows of value matrix
            if parent_node_player == 0:
                world_state_values[:, :, :, parent_node] += (
                        world_state_values[:, :, :, current_node] * (
                    given_strategy[:, :, current_node].reshape(self.number_of_bets, number_of_hands, 1)))

            # else if parent_node player is ip we multiply cols of value matrix
            elif parent_node_player == 1:
                world_state_values[:, :, :, parent_node] += (
                        world_state_values[:, :, :, current_node] * (given_strategy[:, :, current_node]).reshape(
                    self.number_of_bets, 1, number_of_hands))

        return world_state_values

    # # TODO: 4 - possible speed improvement - analyze possibility of vectorization
    # TODO: 1- possible speed improvement - find a way of computing a base for reach probs in main loop of this method

    # 3used
    # create 3D table of all cf values of world nodes, base of almost all cf values and regrets
    def cf_value_world_nodes_table(self, position):
        """ returns (self.number_of_hands)*(self.number_of_hands)*(self.number_of_nodes) numpy array

            create 3D table of all cf values of world nodes by multiplying world nodes cf reach probs and values 3D
             table
         """
        return self.cf_reach_probs_of_world_nodes_table_family(position) * self.values_of_world_nodes_table_family()

    # 2used
    # Here for showing how cf values of info nodes can be computed...you can just use the return formula
    # directly
    def cf_values_of_info_nodes_table_family(self, position):
        """ returns (self.number_of_hands)*(self.number_of_nodes) numpy array """
        return np.sum(self.cf_value_world_nodes_table(position)[:, :, :, :], axis=2 - position)

    # combine two cf_values_of_info_nodes_table(position) for position=0, 1, to one single table containing
    # cf values of info node of to_move player at each node
    def cf_values_of_info_nodes_of_decision_player_table(self):
        """ returns (self.number_of_hands)*(self.number_of_nodes) numpy array """
        cfv_info = np.zeros((self.number_of_bets, self.number_of_hands, self.number_of_nodes))
        for j in self.node:
            p = self.turn[j]
            cfv_info[:, :, j:j + 1] = np.sum(self.cf_value_world_nodes_table(p)[:, :, j], axis=1 - p)[:, np.newaxis]
        return cfv_info

    # 1used
    # important, used in cfr main loop
    def cf_regrets_of_of_info_nodes_table_family(self, position):
        """ returns (self.number_of_hands)*(self.number_of_nodes) numpy array """
        cfr_t = PR = np.zeros((self.number_of_bets, self.number_of_hands, self.number_of_nodes))
        cfv_t = self.cf_values_of_info_nodes_table_family(position).copy()
        nn = self.node[1:]
        for j in nn:
            cfr_t[:, :, j:j + 1] = cfv_t[:, :, j:j + 1] - cfv_t[:, :, self.parent[j]:self.parent[j] + 1]
        return cfr_t

    def cf_regrets_of_public_node_from_world_values_family(self, child):
        node = self.parent[child]
        p = self.turn[node]
        cf_r = (self.values_of_world_nodes_table_family()[:, :, :, child] - self.values_of_world_nodes_table_family()[:,
                                                                            :, :, node]
                ) * self.cf_reach_probs_of_world_nodes_table_family(p)[:, :, :, node]
        return cf_r

    # -------------------------------------------------------------------------------------------------------------------- #
    # 0used
    # so far all the regrets are from op perspective,
    def update_cumulative_regrets(self):
        self.cumulative_regret[:, 0, :, :] += self.cf_regrets_of_of_info_nodes_table_family(0)
        self.cumulative_regret[:, 1, :, :] -= self.cf_regrets_of_of_info_nodes_table_family(1)

    # 0used
    def update_strategy_family(self):
        cr = self.cumulative_regret.copy()
        cr_positive = np.where(cr >= 0, cr, 0)
        for index, d_node in enumerate(self.decision_node):
            turn = self.turn[d_node]
            children = self.decision_node_children[index]
            l = len(children)
            sum_r = np.sum(cr_positive[:, turn, :, children[0]:children[0] + l], axis=-1).reshape(
                self.number_of_bets, self.number_of_hands, 1
            )
            # sum_r_nonzero = np.where(sum_r > 0, sum_r, 1 / l)
            b_sum_r = np.broadcast_to(sum_r, (self.number_of_bets, self.number_of_hands, l))
            self.strategy_base[:, turn, :, children[0]:children[0] + l] = np.divide(
                cr_positive[:, turn, :, children[0]:children[0] + l], b_sum_r,
                out=np.full((self.number_of_bets, self.number_of_hands, l), 1 / l),
                where=(b_sum_r != 0))

    def updated_strategy_family(self):
        strat = np.ones((self.number_of_bets, 2, self.number_of_hands, self.number_of_nodes))
        cr = self.cumulative_regret.copy()
        cr_positive = np.where(cr >= 0, cr, 0)
        for index, d_node in enumerate(self.decision_node):
            turn = self.turn[d_node]
            children = self.decision_node_children[index]
            n_children = len(children)
            sum_r = np.sum(cr_positive[:, turn, :, children[0]:children[0] + n_children], axis=-1)
            for child in self.decision_node_children[index]:
                strat[:, turn, :, child] = cr_positive[:, turn, :, child] / sum_r
        return strat

    # -------------------------------------------------------------------------------------------------------------------- #

    def average_strategy_family(self):
        cum_strat = self.cumulative_strategy.copy()
        avg_strat = np.zeros((self.number_of_bets, 2, self.number_of_hands, self.number_of_nodes))
        for index, d_node in enumerate(self.decision_node):
            children = self.decision_node_children[index]
            l = len(children)
            for p in range(2):
                bros_cum_strat = cum_strat[:, p, :, children[0]:children[0] + l]
                parent_cum_strat = cum_strat[:, p, :, d_node].reshape(self.number_of_bets, self.number_of_hands, 1)
                b_parent_cum_strat = np.broadcast_to(parent_cum_strat, (self.number_of_bets, self.number_of_hands, l))
                avg_strat[:, p, :, children[0]:children[0] + l] = np.divide(
                    bros_cum_strat, b_parent_cum_strat,
                    out=np.zeros((self.number_of_bets, self.number_of_hands, l)),
                    where=(b_parent_cum_strat != 0))
        return avg_strat

    def run_base_cfr(self, number_of_iterations):
        t_start = time.perf_counter()
        for t in range(number_of_iterations):
            self.update_cumulative_regrets()
            self.update_strategy_family()
            self.iteration += 1
        t_finish = time.perf_counter()
        duration = t_finish - t_start
        avg_time_per_1000 = duration / (number_of_iterations / 1000)
        return avg_time_per_1000

    def run_base_cfr_with_progress(self, number_of_iterations):
        t_start = time.perf_counter()
        iter_desc = "CFR With Deck Size equal:  " + str(self.number_of_hands)
        for t in tqdm(range(number_of_iterations), desc=iter_desc):
            self.update_cumulative_regrets()
            self.update_strategy_family()
            self.iteration += 1
        t_finish = time.perf_counter()
        duration = t_finish - t_start
        avg_time_per_1000 = duration / (number_of_iterations / 1000)
        return avg_time_per_1000

# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------- STRATEGY INITIALIZING TOOLS AND SPECIFIC STRATEGIES ---------------------------------- #

    def uniform_strategy_family(self):
        S = np.ones((self.number_of_bets, 2, self.number_of_hands, self.number_of_nodes))
        for _decision_node in self.decision_node:
            childs = self.game.public_state[_decision_node].children
            for child in childs:
                S[:, self.turn[_decision_node], :, child:child + 1] = np.full((
                    self.number_of_bets, self.number_of_hands, 1), 1 / len(childs))
        return S

    def update_strategy_base_to(self, action_prob_function):
        S = np.ones((2, self.number_of_hands, self.number_of_nodes))
        for i in range(self.number_of_hands):
            for _decision_node in self.decision_node:
                childs = self.game.public_state[_decision_node].children
                for child in childs:
                    S[self.turn[_decision_node], i, child:child + 1] = action_prob_function(i, child)
        return S

    def save_current_state(self):
        A = np.zeros((3, 2, self.number_of_hands, self.number_of_nodes))
        A[0:1, :, :, :] = self.initial_strategy.copy()
        A[1:2, :, :, :] = self.cumulative_regret.copy()
        A[2:3, :, :, :] = self.cumulative_strategy
        A[3:4, :, :, :] = self.average_strategy_family()


# ------------------------------- INITIALIZING STRATEGIC BETTING GAMES WITH USUAL SIZES ------------------------------ #
def cfr_family_run_time(max_n_bet, number_of_hands, sub, bet_f=STANDARD_FULL_FAMILY_OF_BETS, iterations=10000,
                        progress_bar=False):
    if progress_bar:
        return StrategyFamilyForCfr(BettingGamesFamily(
            max_n_bet, {i: 1 for i in range(number_of_hands - 1)}, sub, bet_f)).run_base_cfr_with_progress(
            iterations)
    else:
        return StrategyFamilyForCfr(BettingGamesFamily(
            max_n_bet, {i: 1 for i in range(number_of_hands - 1)}, sub, bet_f)).run_base_cfr(iterations)


def hand_family_cfr_family_run_time(max_n_bet, sub, n_hands_family=STANDARD_FULL_FAMILY_OF_HANDS,
                                    bet_f=STANDARD_FULL_FAMILY_OF_BETS, iteration=10000, progress_bar=False):
    if progress_bar:
        return np.array(
            [cfr_family_run_time(max_n_bet, number_of_hands, sub, bet_f, iteration, progress_bar)
             for number_of_hands in tqdm(n_hands_family, desc="Decks")])
    else:
        return np.array(
            [cfr_family_run_time(max_n_bet, number_of_hands, sub, bet_f, iteration, progress_bar)
             for number_of_hands in n_hands_family])


def max_hand_family_cfr_family_run_time(sub, max_family=range(2, 21, 2), n_hands_family=STANDARD_FULL_FAMILY_OF_HANDS,
                                        bet_f=STANDARD_FULL_FAMILY_OF_BETS,
                                        iterations=1000, progress_bar=False):
    matrix = np.zeros((len(max_family), len(n_hands_family)))
    if progress_bar:
        for max_nb in tqdm(range(len(max_family)), desc="Max Number Of Bets "):
            matrix[max_nb, :] = np.array(
                hand_family_cfr_family_run_time(max_family[max_nb], sub, n_hands_family, bet_f, iterations, progress_bar
                                                ))
    else:
        for max_nb in range(len(max_family)):
            matrix[max_nb, :] = np.array(
                hand_family_cfr_family_run_time(max_family[max_nb], sub, n_hands_family, bet_f, iterations, progress_bar
                                                ))
    return matrix


if __name__ == '__main__':
    # t_SFFB_for_2_h3_it1e4 = cfr_family_run_time_with_progress_bar(2, 3, False, 10000)
    # t_SFFB_for_2_h10_it1e4 = cfr_family_run_time_with_progress_bar(2, 10, False, 10000)
    # t_SFFB_for_2_h20_it1e4 = cfr_family_run_time_with_progress_bar(2, 20, False, 10000)
    # t_SFFB_for_2_h40_it1e4 = cfr_family_run_time_with_progress_bar(2, 40, False, 10000)
    # t_SFFB_for_2_h80_it1e4 = cfr_family_run_time_with_progress_bar(2, 80, False, 10000)
    # t_SFFB_for_2_h100_it1e4 = cfr_family_run_time_with_progress_bar(2, 100, False, 10000)

    # ttttt = hand_family_cfr_family_run_time(2, False, n_hands_family=range(3, 10),
    #                                       iteration=100000, progress_bar=True)
    # t_mnb_1e1 = max_hand_family_cfr_family_run_time(False, n_hands_family=range(3, 100),
    #                                                 iterations=100,
    #                                                 bet_f=STANDARD_FULL_FAMILY_OF_BETS,
    #                                                 progress_bar=True)

    ##### Temp2(1)
    # t_mnb__F_2e2 = max_hand_family_cfr_family_run_time(True, n_hands_family=range(3, 100),
    #                                                 iterations=200,
    #                                                 bet_f=STANDARD_FULL_FAMILY_OF_BETS,
    #                                                 progress_bar=True)

    t_mnb__F_1e3 = max_hand_family_cfr_family_run_time(True, n_hands_family=range(3, 100),
                                                       iterations=1000,
                                                       bet_f=STANDARD_FULL_FAMILY_OF_BETS,
                                                       progress_bar=True)



# max2_0_family_times_2to100_1e1 = [cfr_family_run_time(2, n, False, iterations=10) for n in range(3, 101)]
#    T1_100 = max2_0_family_times_2to100_1e1
#
#    max4_0_family_times_2to100_1e1 = [cfr_family_run_time(4, n, False, iterations=10) for n in range(3, 101)]
#
#    hands3_0_family_times_2to100_1e1 = [cfr_family_run_time(n, 3, False, iterations=10) for n in range(2, 13, 2)]
#    hands30_0_family_times_2to100_1e1 = [cfr_family_run_time(n, 30, False, iterations=10) for n in range(2, 13, 2)]
#    hands100_0_family_times_2to100_1e1 = [cfr_family_run_time(n, 100, False, iterations=10) for n in range(2, 13, 2)]
#    T_12_30=[[cfr_family_run_time(n, 10*h, False, iterations=10) for n in range(2, 17, 2)] for h in range(3, 41)]


# kf = BettingGamesFamily(2, {0: 1, 1: 1, 2: 1}, False)
# skf = StrategyFamilyForCfr(kf)
# skf_10000 = skf.run_base_cfr(100000)
# family_av = skf.average_strategy_family()
#
##kf1 = BettingGamesFamily(2, {0: 1, 1: 1, 2: 1}, False)
##skf1 = StrategyFamilyForCfr(kf1)
##skf1_2_000_000 = skf1.run_base_cfr(2000000)
##family_av1 = skf1.average_strategy_family()
#
#
## prb_i_f = skf.players_reach_probs_of_info_nodes_table_with_update_family()
## cf_value_w_0_f = skf.cf_value_world_nodes_table(0)
## cf_value_i_0_f = skf.cf_values_of_info_nodes_table_family(0)
## cf_regret_i_0_f = skf.cf_regrets_of_of_info_nodes_table_family(0)
## skf.update_cumulative_regrets()
#
# av = family_av
# for i in range(32):
#    print(i)
#    print(av[i])
#    print('\n\n')
# np.save('skf_1e5.npy', av)
#
# skf_cum_reg = skf.cumulative_regret
# skf_avg_reg = skf_cum_reg/100000
#
# kf100 = BettingGamesFamily(2, {card: 1 for card in range(100)}, False)
# skf100 = StrategyFamilyForCfr(kf100)
# skf_10000 = skf100.run_base_cfr(1000000)
# family_av100 = skf100.average_strategy_family()



