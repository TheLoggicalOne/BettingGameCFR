import numpy as np


np.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=600, suppress=None, nanstr=None,

                    infstr=None, formatter=None, sign=None, floatmode=None, legacy=None)


def create_sequence_by_multiple_of_power_of_base(base, roof, start_shift=0):
    return np.array([[(start_shift+base ** p) * i for i in range(1, base)] for p in range(1, roof)])


def create_sequence_by_2_5_10(max_power_of_10=10, min_power_of_10=0, initial_points=np.arange(10)):
    points = initial_points
    for n in range(max(min_power_of_10, 1), max_power_of_10):
        by_step_10 = (10 ** (n-1)) * np.arange(10, 20, 1)
        by_step_20 = (10 ** (n-1)) * np.arange(20, 50, 2)
        by_step_50 = (10 ** (n-1)) * np.arange(50, 100, 5)
        points = np.hstack([points, by_step_10, by_step_20, by_step_50])
    return points.T[:, np.newaxis]


def create_sequence_of_numbers(base_step=1, step_multiplier=2, period_length_base=10, period_length_multiplier=2,
                               n_periods=10):
    L = np.zeros((n_periods*period_length_base))
    index = 0
    step = base_step
    for p in range(n_periods):
        for s in range(period_length_base):
            L[index] = L[index-1] + step
            index += 1
        step *= step_multiplier
    return L.reshape(n_periods, period_length_base)



def create_numbers_with_step_sqrt_order(base_step=1, step_multiplier=2, max_multiplier=4, n_periods=10):
    step = base_step
    period_index_from_1 = np.arange(1, n_periods+1)[:, np.newaxis]
    period_index_from_0 = np.arange(n_periods)[:, np.newaxis]
    steps = np.power(step_multiplier, period_index_from_0)
    maxes = np.power(max_multiplier, period_index_from_1)
    mins = np.power(max_multiplier, period_index_from_0)
    n_numbers_in_periods = (maxes - mins)/steps
    return n_numbers_in_periods


#
# if __name__ == '__main__':
#     A = create_sequence_by_2_5_10()
#     A12 = create_sequence_by_2_5_10(12)
#     B = create_sequence_by_2_5_10(10, 2)
#
# geo300 = np.geomspace(1, 1e12, 300)
# geo300_int = geo300.astype(np.int64)
# geo300_int
# geo300[1:]/geo300[0:-1]
#
# np.log(1e12)/np.log(1.1)







