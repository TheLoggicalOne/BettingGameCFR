import numpy as np
import matplotlib.pyplot as plt
from utilities import STANDARD_FULL_FAMILY_OF_BETS, STANDARD_FULL_FAMILY_OF_HANDS
betSFFB_h3to101_m2to21_F_it1e1 = np.load("Time_Family_betSFFB_hand3to101_max2to21_it1e1_subF.npy")
betSFFB_h3to101_m2to21_F_it1e1_2 = np.load("Time_Family_betSFFB_hand3to101_max2to21_it1e1_subF_2.npy")

bet1_h3to100_m2to21_F_it4e3 = np.load("Time_Single_bet1_h3to101_max2to21_it4e3_subF.npy")
bet1_hSFFH_m2to21_T_it2e1 = np.load("Time_Single_bet1_handSFFH_max2to21_it2e1_subT.npy")
bet1_hSFFH_m2to21_F_it2e3 = np.load("Time_Single_bet1_handSFFH_max2to21_it2e3_subF.npy")
bet1_hSFFH_m2to21_T_it3e3 = np.load("Time_Single_bet1_handSFFH_max2to21_it3e3_subT.npy")

F10 = betSFFB_h3to101_m2to21_F_it1e1
S4000 = bet1_h3to100_m2to21_F_it4e3






# a = bet1_hSFFH_m2to21_F_it2e3
# s10 = betSFFB_h3to101_m2to21_F_it1e1

# plt.plot(STANDARD_FULL_FAMILY_OF_HANDS, bet1_hSFFH_m2to21_F_it2e3)
# plt.figure(2)


#strategy_for_cfr_temp5
# Time_Single_bet1_h3to101_max2to21_it4e3_subF = np.load("Time_Single_bet1_handSFFH_max2to21_it2e1_subF.npy")
# np.save("Time_Single_bet1_h3to101_max2to21_it4e3_subF", Time_Single_bet1_h3to101_max2to21_it4e3_subF)
#
# strategy_for_cfr_temp1
# bet1_hSFFH_m2to21_F_it2e3 = np.load("Time_Single_bet1_handSFFH_max2to21_it2e3_subF.npy")




