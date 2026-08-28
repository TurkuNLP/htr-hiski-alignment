import numpy as np


def set_cost(m, cost, char_1, char_2=None):
    if char_2:
        m[ord(char_1), ord(char_2)] = cost
    else:
        m[ord(char_1)] = cost


def set_cost_two_way(m, cost, char_1, char_2):
    set_cost(m, cost, char_1, char_2)
    set_cost(m, cost, char_2, char_1)


def set_cost_from_codes(m, cost, codes_1, codes_2=None):
    if codes_2:
        m[np.ix_(codes_1, codes_2)] = cost
    else:
        m[np.ix_(codes_1)] = cost


def set_cost_two_way_from_codes(m, cost, codes_1, codes_2):
    set_cost_from_codes(m, cost, codes_1, codes_2)
    set_cost_from_codes(m, cost, codes_2, codes_1)
