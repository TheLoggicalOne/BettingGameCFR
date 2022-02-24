import os
import time


import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
import concurrent.futures

from betting_games_for_cfr import BettingGame, BettingGameWorldTree, BettingGamePublicTree
from utilities import MY_FAMILY_OF_BETS, STANDARD_FULL_FAMILY_OF_BETS, STANDARD_FULL_FAMILY_OF_HANDS, SMALL_SAVE_POINTS
from utilities import STANDARD_FULL_FAMILY_OF_SAVE_POINTS_396, STANDARD_REDUCED_FAMILY_OF_SAVE_POINT_266
from tqdm import tqdm
import strategy_for_cfr


def create_StrategyForCfr_game_args_from_file_name(name_str):
    name_parts = name_str.split('_')


    print(name_parts)
    assert name_parts[0] == 'S'
    assert name_parts[1] == 'BG'

    assert name_parts[2][:3] == 'max'
    max_n_bets = int(name_parts[2][3:])

    assert name_parts[3][:4] == 'deck'
    deck_size = int(name_parts[3][4:])

    assert name_parts[4][:3] == 'sub'
    substitution = eval(name_parts[4][3:])

    assert name_parts[5][:3] == 'bet'
    size_of_bet = int(name_parts[5][3:])

    assert name_parts[6] == '.npy'

    # max_n_bet = int(name_str[name_str.index('max'), name_str.index('_deck')])

    return max_n_bets, deck_size, substitution, size_of_bet/100


def get_list_of_files(path=strategy_for_cfr.ARRAYS_PATH):
    return os.listdir(path)


class ListOfNames:

    def __init__(self, list_of_names):
        self.names = list_of_names
        self.list_of_correct_names = []
        self.index_of_correct_names = []
        self.list_of_wrong_names = []
        self.index_of_wrong_names =[]
        for i, name in enumerate(self.names):
            size_of_bet = eval(name.split('_')[5][3:])
            if type(size_of_bet) is int:
                self.list_of_correct_names.append(name)
                self.index_of_correct_names.append(i)
            else:
                self.list_of_wrong_names.append(name)
                self.index_of_wrong_names.append(i)

    def get_list_of_strategy_data_array(self):
        data_names=[]
        for name in self.list_of_correct_names:
            if not name.endswith('points.npy'):
                data_names.append(name)
        return data_names

    def get_StrategyForCfr_game_params(self):
        return [create_StrategyForCfr_game_args_from_file_name(data_name)
                for data_name in self.get_list_of_strategy_data_array()]


def create_StrategyForCfr(betting_game):
    s = strategy_for_cfr.StrategyForCfr(betting_game)
    s.run_vcfr_with_save()


if __name__ == '__main__':
    LIST_OF_ALL_FILES = get_list_of_files()
    NAMES = ListOfNames(LIST_OF_ALL_FILES)
    data_array_names = NAMES.get_list_of_strategy_data_array()
    StrategyForCfr_game_params = NAMES.get_StrategyForCfr_game_params()
    games = [BettingGame(game_params[0], {i: 1 for i in range(game_params[1])}, game_params[2], game_params[3])
             for game_params in StrategyForCfr_game_params]


    tt0 = time.perf_counter()
    processes = []
    for i, b_game in enumerate(games):
        p = multiprocessing.Process(target=create_StrategyForCfr, args=(b_game,))
        p.start()
        processes.append(p)
    #
    for process in processes:
        process.join()
    tt1 = time.perf_counter()
    delta_tt = tt1-tt0





