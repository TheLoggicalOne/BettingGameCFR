# TO BE COMPLETED
import numpy as np
import time
import datetime

np.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=600, suppress=None, nanstr=None,

                    infstr=None, formatter=None, sign=None, floatmode=None, legacy=None)


# Constants
FOLD = 'Fold';  CHECK = 'Check';  CALL = 'Call';  BET = 'Bet';  RAISE = 'Raise'
TERMINAL = 'Terminal';  DECISION = 'Decision';  START = 'Start'
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

_b_small = np.array([int(100 * i / 10) for i in range(21)])
_b_big = 100*np.array([i for i in range(3, 11)])
_b_huge = 100*np.array([27, 81, 729])
_b_huge_s = 100 * np.array([3 ** i for i in range(3, 10)])
_b_huge_sf = 100 * np.array([3 ** i for i in range(3, 10)])
MY_FAMILY_OF_BETS = np.hstack([_b_small, _b_big, _b_huge])



def create_family_of_bets(n_bets, max_big_bets, max_huge_bets):
    bets = np.array([int(100 * i / 10) for i in range(n_bets)])
    big_bets = 100*np.array([i for i in range(3, max_big_bets)])
    huge_bets = 100 * np.array([3 ** i for i in range(3, max_huge_bets)])
    return np.hstack([bets, big_bets, huge_bets])




def create_family_of_hands(small_max, medium_max, big_max, huge_max):
    small_hands = list(range(3, small_max))
    medium_hands = list(range(small_max, medium_max, 5))
    big_hands = list(range(medium_max, big_max, 20))
    huge_max = list(range(big_max, huge_max, 100))
    return np.array(small_hands + medium_hands + big_hands + huge_max)





def create_hands_dealings(OPDeck, IPDeck, substitution=True):
    cards_dealings = []
    if substitution:
        cards_dealings = [(i, j) for i in OPDeck for j in IPDeck]
    else:
        for i in OPDeck:
            for j in IPDeck:
                if j != i:
                    cards_dealings.append((i, j))
    return cards_dealings


def create_points(base, roof, start_shift=0):
    return np.array([[(start_shift+base ** p) * i for i in range(1, base)] for p in range(0, roof)])


def create_sequence_by_2_5_10(max_power_of_10=10, min_power_of_10=1, initial_points=np.arange(10)):
    points = create_points(10, min_power_of_10).flatten()
    points = np.hstack([np.array([0]), points])
    for n in range(max(min_power_of_10, 1), max_power_of_10):
        by_step_10 = (10 ** (n-1)) * np.arange(10, 20, 1)
        by_step_20 = (10 ** (n-1)) * np.arange(20, 50, 2)
        by_step_50 = (10 ** (n-1)) * np.arange(50, 100, 5)
        points = np.hstack([points, by_step_10, by_step_20, by_step_50])
    points = np.hstack([points, np.array([10 ** max_power_of_10])])
    return points.T[:, np.newaxis]


a = create_family_of_bets(21, 10, 10)

STANDARD_FULL_FAMILY_OF_BETS = a

STANDARD_FULL_FAMILY_OF_HANDS = create_family_of_hands(20, 60, 200, 900)

s = create_sequence_by_2_5_10(12)
STANDARD_FULL_FAMILY_OF_SAVE_POINTS_396 = s

sr = create_sequence_by_2_5_10(12, 6)
STANDARD_REDUCED_FAMILY_OF_SAVE_POINT_266 = sr

SMALL_SAVE_POINTS = create_points(10, 6)
s.shape
sr.shape


