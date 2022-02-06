"""
   This is to facilitate working on games and their strategies
   What we need?
   create betting game with given parameters
   create Strategy class of given game with initial strat and run cfr on it for number of iters

   create strategy class for given game, which is characterized by its initial strat and game
   for each strategy, we create npz file that stores info of the given strategy
   self.iteration shows the age of strategy. at each age, we should record its current age, cumulative regret
   and cumulative strategy( initial strategy and game were already given and known)

   So for each vanilla cfr strategy class, which is chracterized by game name and initial strategy:
       (3*2*number_of_hands*number_of_nodes) for cum reg, cum strat, and average strat
       Also possible world node values (number_of_hands*number_of_hands*number_of_nodes)
   This should be done for each iteration (age) and also for different bet sizes

            3 * (2 * M * N)                 cumulative regret and strategy and average strategy
        B * 3 * (2 * M * N)                * number of bet sizes
    S * B * 3 * (2 * M * N)                * number of iterations that we want to save
   For S=30, B=40, M=100, N=100 matrix size will be 72 million and its byte size is 567MB
   M*M*N
G
"""
import math
import time as time
import copy as copy
import numpy as np
import multiprocessing
import concurrent.futures
from betting_games import BettingGame
from strategy import Strategy
from numba import jit, njit, prange

# ---------------------------- CREATING NECESSARY LIST AND ARRAYS AND SAVING THEM  ----------------------------------- #

np.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=300, suppress=None, nanstr=None,

                    infstr=None, formatter=None, sign=None, floatmode=None, legacy=None)

MY_SAVING_RANGES = np.array([2 ** i for i in range(33)])
SMALL_SAVING_RANGE = np.array([2 ** i for i in range(14)])


def saving_range_builder(length):
    return np.array([2 ** i for i in range(length)])


_b_small = np.array([int(100 * i / 10) for i in range(1, 21)])
_b_big = 100*np.array([i for i in range(3, 11)])
_b_huge = 100*np.array([27, 81, 729])
MY_FAMILY_OF_BETS = np.hstack([_b_small, _b_big, _b_huge])


def index_of_bet_in_my_family(bet):
    if bet <= 2:
        return 10*bet-1
    elif 3 <= bet <= 10:
        return bet-2+20-1
    elif bet == 27:
        return 28
    elif bet == 81:
        return 29
    elif bet == 729:
        return 30


STRATEGIES = np.zeros((30, 40, 3, 2, 100, 100))

KF = KUHN_FAMILY_ORIGINAL = [BettingGame(2, {0:1, 1:1, 2:1}, b/100, False) for b in MY_FAMILY_OF_BETS]
SKF = [Strategy(k_game) for k_game in KF]
KF_ARRAY_NAMES = []
for ii in range(len(MY_FAMILY_OF_BETS)):
    np.save('V'+'_'+SKF[ii].numpy_file_str, np.zeros((len(MY_SAVING_RANGES), 3, 2, 3, 9)))
    KF_ARRAY_NAMES.append('V'+'_'+SKF[ii].numpy_file_str)

V_B_2_3_0_F = np.zeros((len(MY_FAMILY_OF_BETS), len(MY_SAVING_RANGES), 3, 2, 3, 9))
np.save('V_B_2_3_0_F', V_B_2_3_0_F)


def create_game_from_str_name(game_name):
    ps = [int(part) for part in game_name.split('_')[1:]]
    return BettingGame(ps[0], deck={i: 1 for i in range(ps[1])}, bet_size=ps[3]/100,
                       deal_from_deck_with_substitution=bool(ps[2]))

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def run_vcfr_with_save(strat, saving_points=MY_SAVING_RANGES, start_age_index=0):
    bet = strat.game.bet_size
    index_bet = int(index_of_bet_in_my_family(bet))
    for index_age in range(start_age_index, len(saving_points) - 1):       # i in [0, 1, ...., 32]
        V_B_2_3_0_F = np.load('V_B_2_3_0_F.npy')
        # for j in range(saving_points[index_age], saving_points[index_age+1]):
        strat.run_base_cfr(saving_points[index_age+1]-saving_points[index_age])
        V_B_2_3_0_F[index_bet, index_age, 0, :, :, :] = strat.cumulative_regret.copy()
        V_B_2_3_0_F[index_bet, index_age, 1, :, :, :] = strat.cumulative_strategy.copy()
        V_B_2_3_0_F[index_bet, index_age, 2, :, :, :] = strat.average_strategy().copy()
        np.save('V_B_2_3_0_F', V_B_2_3_0_F)
    return f"ran vcfr for {strat.game.name}"


def run_vcfr_with_save_separate(strat, saving_points=MY_SAVING_RANGES, start_age_index=0):
    bet = strat.game.bet_size
    index_bet = int(index_of_bet_in_my_family(bet))
    for index_age in range(start_age_index, len(saving_points) - 1):       # i in [0, 1, ...., 32]
        V = np.load(KF_ARRAY_NAMES[index_bet])
        # for j in range(saving_points[index_age], saving_points[index_age+1]):
        strat.run_base_cfr(saving_points[index_age+1]-saving_points[index_age])
        V[index_age, 0, :, :, :] = strat.cumulative_regret.copy()
        V[index_age, 1, :, :, :] = strat.cumulative_strategy.copy()
        V[index_age, 2, :, :, :] = strat.average_strategy().copy()
        np.save(KF_ARRAY_NAMES[index_bet], V)
    print(f"ran vcfr for {strat.game.name}")


def run_vcfr_small_save(strat):
    return run_vcfr_with_save(strat, saving_points=SMALL_SAVING_RANGE)


def run_vcfr_family_with_save(strat_family, saving_points=MY_SAVING_RANGES):
    for strat_index in prange(len(strat_family)):
        run_vcfr_with_save(strat_family[strat_index], saving_points)


def run_vcfr_family_with_save_separate(strat_family, saving_points=MY_SAVING_RANGES):
    for strat_index in prange(len(strat_family)):
        run_vcfr_with_save_separate(strat_family[strat_index], saving_points)

# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------- PARALLELIZATION SEPARATE ----------------------------------------------- #


tt0 = time.perf_counter()
processes = []
for b in range(len(MY_FAMILY_OF_BETS)):
    p = multiprocessing.Process(target=run_vcfr_with_save_separate, args=(SKF[b], saving_range_builder(17), 0))
    p.start()
    processes.append(p)
#
for process in processes:
    process.join()
tt1 = time.perf_counter()
delta_tt = tt1-tt0
#


# with concurrent.futures.ProcessPoolExecutor() as executor:
#     results = executor.map(run_vcfr_samll_save, SKF)

# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------ PARALLELIZATION --------------------------------------------------- #
#processes = []
#for b in range(len(MY_FAMILY_OF_BETS)):
#    p = multiprocessing.Process(target=run_vcfr_with_save, args=(SKF[b], SMALL_SAVING_RANGE, 0))
#    p.start()
#    processes.append(p)
#
#for process in processes:
#    process.join()
#


# with concurrent.futures.ProcessPoolExecutor() as executor:
#     results = executor.map(run_vcfr_samll_save, SKF)

# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------- INITIALIZING SOME EXAMPLES --------------------------------------------- #
#t0 = time.perf_counter()
#run_vcfr_with_save(SKF[4], SMALL_SAVING_RANGE, 0)
#t1 = time.perf_counter()
#time_delta = t1 - t0
#VR = np.load('V_B_2_3_0_F.npy')



# t2 = time.perf_counter()
#
# run_vcfr_family_with_save(SKF, saving_points=SMALL_SAVING_RANGE)
#
# t3 = time.perf_counter()
# time_delta_1 = t3 - t2


#t4 = time.perf_counter()
##
#run_vcfr_family_with_save_separate(SKF, saving_points=SMALL_SAVING_RANGE)
##
#t5 = time.perf_counter()
#time_delta_2 = t5 - t4
#
#V50 = np.load(KF_ARRAY_NAMES[4])
#V10 = np.load(KF_ARRAY_NAMES[0])
#V200 = np.load(KF_ARRAY_NAMES[19])
