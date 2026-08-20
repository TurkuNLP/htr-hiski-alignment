import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def levenshtein_distance_extended(
    str string_1,
    str string_2,
    cnp.ndarray[double, ndim=2] substitute_costs,
    cnp.ndarray[double, ndim=1] insert_costs = None,
    cnp.ndarray[double, ndim=1] delete_costs = None,
    dict deabbreviate_costs = None,
    double deabbreviate_penalty = 1.0,
):
    cdef int m = len(string_1)
    cdef int n = len(string_2)
    cdef cnp.ndarray[double, ndim=2] dp = np.zeros(shape=(m + 1, n + 1))

    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)

    cdef int i, j
    cdef Py_UCS4 char_1, char_2
    cdef int char_1_code, char_2_code
    cdef double substitute_cost, insert_cost, delete_cost, deabbreviate_cost, min_cost
    cdef cnp.ndarray[double, ndim=1] deabbreviate_continue_penalties = deabbreviate_penalty * np.arange(1, n + 1)

    for i in range(1, m + 1):
        char_1 = string_1[i - 1]
        if deabbreviate_costs is not None and char_1 in deabbreviate_costs:
            deabbreviate_cost = deabbreviate_costs[char_1]
            dp[i, 1:] = (
                i * deabbreviate_cost * deabbreviate_continue_penalties
            )
            continue

        for j in range(1, n + 1):
            char_2 = string_2[j - 1]
            char_1_code = ord(char_1)
            char_2_code = ord(char_2)

            substitute_cost = substitute_costs[char_1_code, char_2_code]
            delete_cost = (
                delete_costs[char_1_code]
                if delete_costs is not None
                else 1.0
            )
            insert_cost = (
                insert_costs[char_2_code]
                if insert_costs is not None
                else 1.0
            )

            min_cost = dp[i - 1, j] + delete_cost
            if dp[i, j - 1] + insert_cost < min_cost:
                min_cost = dp[i, j - 1] + insert_cost
            if dp[i - 1, j - 1] + substitute_cost < min_cost:
                min_cost = dp[i - 1, j - 1] + substitute_cost
            dp[i, j] = min_cost

    return dp[m, n]

def lde_similarity(
    str string_1,
    str string_2,
    cnp.ndarray[double, ndim=2] substitute_costs,
    cnp.ndarray[double, ndim=1] insert_costs = None,
    cnp.ndarray[double, ndim=1] delete_costs = None,
    dict deabbreviate_costs = None,
    double deabbreviate_penalty = 1.0
):
    cdef double edit_distance
    cdef int max_len

    edit_distance = levenshtein_distance_extended(
        string_1,
        string_2,
        substitute_costs,
        insert_costs,
        delete_costs,
        deabbreviate_costs,
        deabbreviate_penalty
    )

    cdef int len_1 = len(string_1)
    cdef int len_2 = len(string_2)
    if len_1 >= len_2:
        max_len = len_1
    else:
        max_len = len_2

    return 1 - (edit_distance / max_len)
