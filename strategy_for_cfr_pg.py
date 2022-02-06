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
import os
import time

import numpy
import numpy as np
from betting_games_for_cfr import BettingGame, BettingGameWorldTree, BettingGamePublicTree
from utilities import START_NODE, MY_FAMILY_OF_BETS, create_points
import matplotlib.pyplot as plt
from tqdm import tqdm



np.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=600, suppress=None, nanstr=None,

                    infstr=None, formatter=None, sign=None, floatmode=None, legacy=None)

n_of_saves_for_single_bet = 10000
n_of_saves = n_of_saves_for_single_bet
ssr = 5
SMALL_SAVE_POINTS = create_points(10, 5)



# Where to save the arrays
PROJECT_ROOT_DIR = "."
MODULE_ARRAY_PATH = "strategy_for_cfr_arrays"
ARRAYS_PATH = os.path.join(PROJECT_ROOT_DIR, "numpy_arrays_dir", MODULE_ARRAY_PATH)
os.makedirs(ARRAYS_PATH, exist_ok=True)


class StrategyForCfr:

    def __init__(self, game, initial_strategy=None, naming_key='', saving_points=SMALL_SAVE_POINTS):
        self.game = game
        self.naming_key = naming_key
        self.number_of_hands = self.game.number_of_hands
        self.number_of_nodes = self.game.public_tree.number_of_nodes
        self.saving_points = saving_points

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

        self.terminal_values_all_nodes = self.game.terminal_values_all_nodes.copy()

        self.chance_reach_prob = self.game.deck_matrix()

        self.name = 'S' + '_' + self.game.name + '_' + self.naming_key



        self.strategy_data_array_path = os.path.join(ARRAYS_PATH, self.name + '.npy')
        self.save_points_array_path = os.path.join(ARRAYS_PATH, self.name + '_saves_points' + '.npy')

        # fucking steps:

        # no matter what case has happened above load corresponding saved arrays to 2 attrs of class
        # populate related strategt attrs using this 2 arrays
        # note that  self.strategy_data_path exists if and only if self.save_points_file_path exists

        # check if there are numpy arrays on paths
        # if no array save appropriate arrays, all with zeros
        if not os.path.exists(self.strategy_data_array_path):
            if initial_strategy is None:
                initial_strategy = self.uniform_strategy()
                self.initial_strategy = initial_strategy.copy()
                self.strategy_base = self.initial_strategy.copy()

            if type(initial_strategy) is np.ndarray:

                np.save(self.strategy_data_array_path,
                        np.zeros((n_of_saves, 3, 2, self.number_of_hands, self.number_of_nodes)))
                np.save(self.save_points_array_path, np.zeros((n_of_saves, 4)))
            else:
                np.save(self.strategy_data_array_path, initial_strategy.strategy_data_array)
                np.save(self.save_points_array_path, initial_strategy.save_points_array)


        self.strategy_data_array = np.load(self.strategy_data_array_path)
        #strategy_data_array is n_of_saves*3*(2*number_of_hands*number_of_nodes) where at each save points, 3 arrays
        # with dimensions of (2*number_of_hands*number_of_nodes) corresponding to cum_reg and cum_strat and
        # average_strat

        self.save_points_array = np.load(self.save_points_array_path )
        # save_points_array is n*4 np.array that its first row is in order: last filled index, ,
        # last saved iteration, Total time, heritrage code and other rows of 4 cols are in order:
        # save points, time.perf_counter, time.process_time(), time returned by run_base_cfr

        self.last_saved_index_in_saved_arrays_1d0d = self.save_points_array[0:1, 0]
        self.age_1d0d = self.save_points_array[0:1, 1]
        self.iteration = 0


        self.cumulative_strategy = self.strategy_data_array[int(self.last_saved_index_in_saved_arrays_1d0d[0]),
                                                            0, :, :, :].copy()
        self.cumulative_regret = self.strategy_data_array[int(self.last_saved_index_in_saved_arrays_1d0d[0]),
                                                            1, :, :, :].copy()
        if type(initial_strategy) is np.ndarray:
            self.update_cumulative_regrets()

            # self.update_strategy()
            # self.iteration += 1


# case1) start from numpy.ndarray:
#
#
# case2) start from strategy with array records





# ------------------------------- SAVING AND READING INTO NUMPY ARRAYS NPZ FILE -------------------------------------- #
    def sync_from_strategy_data(self, save_point=None):
        arr_points = np.load(self.save_points_array_path)
        arr_data = np.load(self.strategy_data_array_path)
        if save_point is None:
            self.cumulative_regret = arr_data









# ---------------------------- MAIN METHODS: REACH PROBABILITIES OF GIVEN STRATEGY  ---------------------------------- #

    # 9used

    # even cols are none-one op cols

    def check_branch_strategy(self, position):
        """ return columns corresponding to decision nodes of check branch of game, in strategy matrix of given player

        First column of each of two table corresponds to column 1 in strategy_base
        even indexed columns are none-one op cols """
        return np.hstack((self.strategy_base[position, :, 1:2],
                          self.strategy_base[position, :, 4:self.check_decision_branch[-1] + 1:6]))

    # 9used
    # odd cols are none-one op cols
    def bet_branch_strategy(self, position):
        """ return columns corresponding to decision nodes of bet branch of game, in strategy matrix of given player

        First column of each of two table corresponds to  column 2 in strategy_base
        even indexed columns are none-one op cols """
        return np.hstack((self.strategy_base[position, :, 2:3],
                          self.strategy_base[position, :, 7:self.bet_decision_branch[-1] + 1:6]))

    # 8used
    def player_reach_probs_of_check_decision_branch_info_nodes(self, position):
        """ return reach probability columns of decision nodes in check branch of game for given player

        First column corresponds of each player table to to column 1 in strategy_base  """
        return np.cumprod(self.check_branch_strategy(position), axis=1)

    # 8used
    def player_reach_probs_of_bet_decision_branch_info_nodes(self, position):
        """ return reach probability columns of decision nodes in bet branch of game for given player

               First column corresponds of each player table to to column 2 in strategy_base  """
        return np.cumprod(self.bet_branch_strategy(position), axis=1)

    # 7used
    # This is the main calculator of  player reach probs of given info node, vectorized over hands of player
    def player_reach_probs_of_info_node(self, node, position):
        """ returns (self.number_of_hands)*1 numpy array where row i corresponds to info node(hand=i, node)"""
        if node == 0:
            PRN = np.ones((self.number_of_hands, 1))
        elif self.start_with_check[node]:
            if self.is_decision_node[node]:
                return self.player_reach_probs_of_check_decision_branch_info_nodes(position)[:,
                       self.depth_of_node[node] - 1:self.depth_of_node[node]]
            else:
                parent = self.parent[node]
                PRN = self.strategy_base[position, :,
                      node:node + 1] * self.player_reach_probs_of_check_decision_branch_info_nodes(
                    position)[:, self.depth_of_node[parent] - 1:self.depth_of_node[parent]]

        else:
            if self.is_decision_node[node]:
                PRN = self.player_reach_probs_of_bet_decision_branch_info_nodes(position)[:,
                      self.depth_of_node[node] - 1:self.depth_of_node[node]]
            else:
                parent = self.parent[node]
                PRN = self.strategy_base[position, :, node:node + 1] \
                      * self.player_reach_probs_of_bet_decision_branch_info_nodes(position)[:,
                        self.depth_of_node[parent] - 1:self.depth_of_node[parent]]
        return PRN

    # 6used
    # This just tabularize player_reach_probs_of_info_node method
    # makes 2 table one table for each position, each table has one column( size of number of hands) for each node
    def players_reach_probs_of_info_nodes_table_with_update(self):
        """ returns 2*(self.number_of_hands)*(self.number_of_nodes) numpy array """
        PR = np.ones((2, self.number_of_hands, self.number_of_nodes))
        for i in range(2):
            for node in self.node[1:]:
                PR[i, :, node:node + 1] = self.player_reach_probs_of_info_node(node, i)
        # This part update self.cumulative_strategy
        self.cumulative_strategy += PR
        return PR

    # TODO: each time update_cumulative_regrets is called, this will be called 6 level deep seems like best place
    #  to update strategy sum is to do it inside this method!

    def players_reach_probs_of_info_nodes_table(self):
        """ returns 2*(self.number_of_hands)*(self.number_of_nodes) numpy array """
        PR = np.ones((2, self.number_of_hands, self.number_of_nodes))
        for i in range(2):
            for node in self.node[1:]:
                PR[i, :, node:node + 1] = self.player_reach_probs_of_info_node(node, i)
        return PR

    # 4used
    # Create 3D table,o for given player, containing cf reach probs( opponent reach probs) of world nodes
    # world node reach probs for each player are equal to info node reach probs
    # creation is by broadcasting player_reach_probs_of_info_nodes_table for one player, by repeating it for each
    # opponent possible hand
    def cf_reach_probs_of_world_nodes_table(self, position):
        """ returns (self.number_of_hands)*(self.number_of_hands)*(self.number_of_nodes) numpy array """

        if position == 0:
            reach_of_info_nodes_table = self.players_reach_probs_of_info_nodes_table_with_update()[1:2, :, :]
            OR = np.broadcast_to(reach_of_info_nodes_table,
                                 (self.number_of_hands, self.number_of_hands, self.number_of_nodes))
        else:
            reach_of_info_nodes_table = self.players_reach_probs_of_info_nodes_table_with_update()[0:1, :, :].reshape(
                self.number_of_hands, 1, self.number_of_nodes)
            OR = np.broadcast_to(reach_of_info_nodes_table,
                                 (self.number_of_hands, self.number_of_hands, self.number_of_nodes))
        return OR

    # TODO: 4 - possible speed improvement - analyze possibility of better vectorization

    # create 3D table containing, tabular version of reach_prob_of_world_node
    # 3D array populate element by element, no vectorization so far
    def reach_probs_of_world_nodes_table(self):
        """ returns (self.number_of_hands)*(self.number_of_hands)*(self.number_of_nodes) numpy array """
        R = np.ones((self.number_of_hands, self.number_of_hands, self.number_of_nodes))
        for op_hand in range(self.number_of_hands):
            for ip_hand in range(self.number_of_hands):
                for node in self.game.node[1:]:
                    R[op_hand, ip_hand, node,] = self.reach_prob_of_world_node(node, [op_hand, ip_hand])
        return R

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
    def values_of_world_nodes_table(self):
        """ returns (self.number_of_hands)*(self.number_of_hands)*(self.number_of_nodes) numpy array """

        number_of_hands = self.number_of_hands
        world_state_values = self.terminal_values_all_nodes.copy()
        given_strategy = self.strategy_base[0, :, :] * self.strategy_base[1, :, :]
        # world_state_value[ , , t.parent] += world_state_value[ , , t]**strategy[ , t]
        nonzero_nodes = self.node[1:]
        reverse_node_list = nonzero_nodes[::-1]

        for current_node in reverse_node_list:
            current_player = self.turn[current_node]
            parent_node = self.parent[current_node]
            parent_node_player = 1 - current_player

            # if parent_node player is op we multiply rows of value matrix
            if parent_node_player == 0:
                world_state_values[:, :, parent_node] += (
                        world_state_values[:, :, current_node] * (
                    given_strategy[:, current_node].reshape(number_of_hands, 1)))

            # else if parent_node player is ip we multiply cols of value matrix
            elif parent_node_player == 1:
                world_state_values[:, :, parent_node] += (
                        world_state_values[:, :, current_node] * given_strategy[:, current_node])

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
        return self.cf_reach_probs_of_world_nodes_table(position) * self.values_of_world_nodes_table()

    # 2used
    # Here for showing how cf values of info nodes can be computed...you can just use the return formula
    # directly
    def cf_values_of_info_nodes_table(self, position):
        """ returns (self.number_of_hands)*(self.number_of_nodes) numpy array """
        return np.sum(self.cf_value_world_nodes_table(position)[:, :, :], axis=1 - position)

    # combine two cf_values_of_info_nodes_table(position) for position=0, 1, to one single table containing
    # cf values of info node of to_move player at each node
    def cf_values_of_info_nodes_of_decision_player_table(self):
        """ returns (self.number_of_hands)*(self.number_of_nodes) numpy array """
        cfv_info = np.zeros((self.number_of_hands, self.number_of_nodes))
        for j in self.node:
            p = self.turn[j]
            cfv_info[:, j:j + 1] = np.sum(self.cf_value_world_nodes_table(p)[:, :, j], axis=1 - p)[:, np.newaxis]
        return cfv_info

    # 1used
    # important, used in cfr main loop
    def cf_regrets_of_of_info_nodes_table(self, position):
        """ returns (self.number_of_hands)*(self.number_of_nodes) numpy array """
        cfr_t = PR = np.zeros((self.number_of_hands, self.number_of_nodes))
        cfv_t = self.cf_values_of_info_nodes_table(position).copy()
        nn = self.node[1:]
        for j in nn:
            cfr_t[:, j:j + 1] = cfv_t[:, j:j + 1] - cfv_t[:, self.parent[j]:self.parent[j] + 1]
        return cfr_t

    def cf_regrets_of_public_node_from_world_values(self, child):
        node = self.parent[child]
        p = self.turn[node]
        cf_r = (self.values_of_world_nodes_table()[:, :, child] - self.values_of_world_nodes_table()[:, :, node]
                ) * self.cf_reach_probs_of_world_nodes_table(p)[:, :, node]
        return cf_r

# ------------------------------------------UPDATING REGRETS AND STRATEGY--------------------------------------------- #
    # 0used
    # so far all the regrets are from op perspective,
    def update_cumulative_regrets(self):
        self.cumulative_regret[0, :, :] += self.cf_regrets_of_of_info_nodes_table(0)
        self.cumulative_regret[1, :, :] -= self.cf_regrets_of_of_info_nodes_table(1)

    # 0used
    def update_strategy(self):
        cr = self.cumulative_regret.copy()
        cr_positive = np.where(cr >= 0, cr, 0)
        for index, d_node in enumerate(self.decision_node):
            turn = self.turn[d_node]
            children = self.decision_node_children[index]
            l = len(children)
            sum_r = np.sum(cr_positive[turn, :, children[0]:children[0] + l], axis=1)[:, np.newaxis]
            # sum_r_nonzero = np.where(sum_r > 0, sum_r, 1 / l)
            b_sum_r = np.broadcast_to(sum_r, (self.number_of_hands, l))
            self.strategy_base[turn, :, children[0]:children[0] + l] = np.divide(
                cr_positive[turn, :, children[0]:children[0] + l], b_sum_r,
                out=np.full((self.number_of_hands, l), 1 / l),
                where=(b_sum_r != 0))

    def updated_strategy(self):
        strat = np.ones((2, self.number_of_hands, self.number_of_nodes))
        cr = self.cumulative_regret.copy()
        cr_positive = np.where(cr >= 0, cr, 0)
        for index, d_node in enumerate(self.decision_node):
            turn = self.turn[d_node]
            children = self.decision_node_children[index]
            n_children = len(children)
            sum_r = np.sum(cr_positive[turn, :, children[0]:children[0] + n_children], axis=1)
            for child in self.decision_node_children[index]:
                strat[turn, :, child] = cr_positive[turn, :, child] / sum_r
        return strat

# -------------------------------------------------RUN ITERATIONS----------------------------------------------------- #

    def average_strategy(self):
        cum_strat = self.cumulative_strategy.copy()
        avg_strat = np.zeros((2, self.number_of_hands, self.number_of_nodes))
        for index, d_node in enumerate(self.decision_node):
            children = self.decision_node_children[index]
            l = len(children)
            for p in range(2):
                bros_cum_strat = cum_strat[p, :, children[0]:children[0] + l]
                parent_cum_strat = cum_strat[p, :, d_node][:, np.newaxis]
                b_parent_cum_strat = np.broadcast_to(parent_cum_strat, (self.number_of_hands, l))
                avg_strat[p, :, children[0]:children[0] + l] = np.divide(
                    bros_cum_strat, b_parent_cum_strat,
                    out=np.zeros((self.number_of_hands, l)),
                    where=(b_parent_cum_strat != 0))
        return avg_strat

    def run_base_cfr(self, number_of_iterations):
        t_start = time.perf_counter()
        for t in range(number_of_iterations):
            self.update_cumulative_regrets()
            self.update_strategy()
            self.iteration += 1
        t_finish = time.perf_counter()
        duration = t_finish - t_start
        avg_time_per_1000 = duration / (number_of_iterations / 1000)
        return avg_time_per_1000

    def run_base_cfr_with_progress(self, number_of_iterations):
        t_start = time.perf_counter()
        for t in tqdm(range(number_of_iterations)):
            self.update_cumulative_regrets()
            self.update_strategy()
            self.iteration += 1
        t_finish = time.perf_counter()
        duration = t_finish - t_start
        avg_time_per_1000 = duration / (number_of_iterations / 1000)
        return avg_time_per_1000

    def run_vcfr_with_save(self, saving_points=None, start_age_index=0):
        if saving_points is None:
            saving_points = self.saving_points
        bet = self.game.bet_size
        for index_age in range(start_age_index, len(saving_points) - 1):  # i in [0, 1, ...., 32]
            V = np.load(KF_ARRAY_NAMES[index_bet])
            # for j in range(saving_points[index_age], saving_points[index_age+1]):
            self.run_base_cfr(saving_points[index_age + 1] - saving_points[index_age])
            V[index_age, 0, :, :, :] = self.cumulative_regret.copy()
            V[index_age, 1, :, :, :] = self.cumulative_strategy.copy()
            V[index_age, 2, :, :, :] = self.average_strategy().copy()
            np.save(KF_ARRAY_NAMES[index_bet], V)
        print(f"ran vcfr for {self.game.name}")


# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------- STRATEGY INITIALIZING TOOLS AND SPECIFIC STRATEGIES ---------------------------------- #

    def uniform_strategy(self):
        S = np.ones((2, self.number_of_hands, self.number_of_nodes))
        for _decision_node in self.decision_node:
            childs = self.game.public_state[_decision_node].children
            for child in childs:
                S[self.turn[_decision_node], :, child:child + 1] = np.full((
                    self.number_of_hands, 1), 1 / len(childs))
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
        A[3:4, :, :, :] = self.average_strategy()


# ------------------------------- INITIALIZING STRATEGIC BETTING GAMES WITH USUAL SIZES ------------------------------ #

def create_family_of_bets(n_bets, max_big_bets, max_huge_bets):
    bets = np.array([int(100 * i / 10) for i in range(n_bets)])
    big_bets = 100 * np.array([i for i in range(3, max_big_bets)])
    huge_bets = 100 * np.array([3 ** i for i in range(3, max_huge_bets)])
    return np.hstack([bets, big_bets, huge_bets])


STANDARD_FULL_FAMILY_OF_BETS = create_family_of_bets(21, 10, 10)


def create_family_of_hands(small_max, medium_max, big_max, huge_max):
    small_hands = list(range(3, small_max))
    medium_hands = list(range(small_max, medium_max, 5))
    big_hands = list(range(medium_max, big_max, 20))
    huge_max = list(range(big_max, huge_max, 100))
    return np.array(small_hands + medium_hands + big_hands + huge_max)


STANDARD_FULL_FAMILY_OF_HANDS = create_family_of_hands(20, 60, 200, 900)


def bet_family_cfr_run_time(max, number_of_hands, sub, bet_family=MY_FAMILY_OF_BETS, iterations=10):
    bf = [BettingGame(max, {i: 1 for i in range(number_of_hands)}, sub, b / 100) for b in bet_family]
    sbf = [StrategyForCfr(bf[i]) for i in range(len(bet_family))]
    rbf = [sbf[i].run_base_cfr(iterations) for i in range(len(bet_family))]
    return rbf


def bet_hand_family_cfr_run_time(max, sub, n_hands_family=range(3, 101), bet_family=MY_FAMILY_OF_BETS, iterations=10):
    matrix = np.zeros((len(n_hands_family), len(bet_family)))
    for nh_index in range(len(n_hands_family)):
        matrix[nh_index, :] = np.array(
            bet_family_cfr_run_time(max, n_hands_family[nh_index], sub, bet_family, iterations))
    return matrix


def max_bet_hand_family_cfr_run_time(sub, max_family=range(2, 21, 2), n_hands_family=range(3, 101),
                                     bet_family=MY_FAMILY_OF_BETS, iterations=10):
    matrix = np.zeros((len(max_family), len(n_hands_family), len(bet_family)))
    for max_index in range(len(max_family)):
        matrix[max_index, :, :] = np.array(
            bet_hand_family_cfr_run_time(max_family[max_index], sub, n_hands_family, bet_family, iterations)
        )
    return matrix


if __name__ == '__main__':
    KUHN_ORIG = BettingGame(2, {i: 1 for i in range(3)}, False, 1)
    KUHN_ORIG_STRATEGY = StrategyForCfr(KUHN_ORIG)
    sb = np.load(KUHN_ORIG_STRATEGY.strategy_data_array_path)
# kost = KUHN_ORIG_STRATEGY.run_base_cfr_with_progress(100000)
# t_of_h3to100_for_2_f = bet_hand_family_cfr_run_time(2, False, range(3, 101), range(1, 2), 1000)
# t_of_hSFFH_for_2_f = bet_hand_family_cfr_run_time(2, False, STANDARD_FULL_FAMILY_OF_HANDS, range(1, 2), 1000)
# plt.plot(STANDARD_FULL_FAMILY_OF_HANDS, t_of_hSFFH_for_2_f)
# plt.figure(2)
# plt.plot(range(3, 101), t_of_h3to100_for_2_f)
#
# t_of_h3to100_for_2_f_it1e = bet_hand_family_cfr_run_time(2, False, range(3, 101), STANDARD_FULL_FAMILY_OF_BETS,
#                                                        1000000)
#
# t_of_max_and_hand_FFS_it_10 = max_bet_hand_family_cfr_run_time(False,
#                                                            max_family=range(2, 21, 2),
#                                                            n_hands_family=STANDARD_FULL_FAMILY_OF_HANDS,
#                                                            bet_family=range(1,2),
#                                                            iterations=10)
#
#   max2_0_family_times_1to20_1e4 = [family_cfr_run_time(2, n, False) for n in range(2, 12)]
#
# frt100_1e3 = bet_family_cfr_run_time(2, 100, False, iterations=1000)
# frt10_1e1 = bet_family_cfr_run_time(2, 10, False, iterations=10)
#
# nhfamily_10to100_by10_betfamily_iter_1e1 = bet_hand_family_cfr_run_time(2, False, range(10, 101, 10))
# FFRT = nhfamily_10to100_by10_betfamily_iter_1e1
#
# nhfamily_max2_10to100_by10_betfamily_iter_1e1 = bet_hand_family_cfr_run_time(4, False, range(10, 101, 10))
# FFRT4 = nhfamily_max2_10to100_by10_betfamily_iter_1e1
#
# nhfamily_max6_10to100_by10_betfamily_iter_1e1 = bet_hand_family_cfr_run_time(6, False, range(10, 101, 10))
# FFRT6 = nhfamily_max6_10to100_by10_betfamily_iter_1e1
#
# standard_times_iter_10 = max_bet_hand_family_cfr_run_time(False,
#                                                           n_hands_family=STANDARD_FULL_FAMILY_OF_HANDS,
#                                                           bet_family=STANDARD_FULL_FAMILY_OF_BETS)

#    frt100_1e2 = family_cfr_run_time(2, 200, False, iterations=100)
#    frt200_1e2 = frt100_1e2
#    #k = BettingGame(2, {0: 1, 1: 1, 2: 1}, False, 0.5)
#    #sk = StrategyForCfr(k)
#    #r = sk.run_base_cfr(100000)
#    #k60 = BettingGame(2, {0: 1, 1: 1, 2: 1}, False, 0.6)
#    #sk60 = StrategyForCfr(k60)
#    #r60 = sk60.run_base_cfr(100000)
#
#    kf = [BettingGame(2, {0: 1, 1: 1, 2: 1}, False, b/100) for b in MY_FAMILY_OF_BETS]
#    skf = [StrategyForCfr(kf[i])for i in range(len(MY_FAMILY_OF_BETS))]
#
#    t0 = time.perf_counter()
#    rkf = [skf[i].run_base_cfr(100000) for i in range(len(MY_FAMILY_OF_BETS))]
#    delta = time.perf_counter()-t0
#
#    avkf = [skf[i].average_strategy() for i in range(len(MY_FAMILY_OF_BETS))]
#
#    av = avkf
#    for i in range(32):
#        print(i)
#        print(av[i])
#        print('\n\n')
#
#    avnp = np.array(av)
#
#    skf_1e5 = np.load('skf_1e5.npy')
#
#    np.array_equal(skf_1e5, av)
