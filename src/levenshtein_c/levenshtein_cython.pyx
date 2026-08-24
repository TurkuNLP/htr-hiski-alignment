# distutils: language = c++
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport pow
from libc.stdint cimport uint32_t
from libcpp.unordered_set cimport unordered_set

@cython.boundscheck(False)
@cython.wraparound(False)
def levenshtein_distance_extended(
    str string_1,
    str string_2,
    cnp.ndarray[double, ndim=2] substitute_costs,
    cnp.ndarray[double, ndim=1] insert_costs,
    cnp.ndarray[double, ndim=1] delete_costs,
    dict deabbreviate_costs,
    double deabbreviate_penalty = 1.0,
    int deabbreviate_max_len = 0
):
    cdef int m = len(string_1)
    cdef int n = len(string_2)

    cdef cnp.ndarray[double, ndim=2] dp = np.zeros(shape=(m + 1, n + 1))

    cdef int i, j, k, l
    cdef Py_UCS4 char_1
    cdef int char_1_code, char_2_code
    cdef double substitute_cost, insert_cost, delete_cost, deabbreviate_cost, min_cost, span_cost
    cdef cnp.ndarray[double, ndim=1] char_1_deabbreviate_costs
    cdef cnp.ndarray[Py_UCS4, ndim=1] string_1_char_codes = cnp.ndarray(m, dtype=np.uint32)
    cdef cnp.ndarray[Py_UCS4, ndim=1] string_2_char_codes = cnp.ndarray(n, dtype=np.uint32)
    cdef unordered_set[uint32_t] deabbreviate_char_codes
    for key in deabbreviate_costs.keys():
        deabbreviate_char_codes.insert(ord(key))

    for i in range(m + 1):
        dp[i, 0] = i
    for i in range(m):
        string_1_char_codes[i] = ord(
            string_1[i]
        )
    for j in range(n + 1):
        dp[0, j] = j
    for j in range(n):
        string_2_char_codes[j] = ord(
            string_2[j]
        )

    for i in range(1, m + 1):
        char_1 = string_1[i - 1]
        char_1_code = string_1_char_codes[i - 1]

        if deabbreviate_char_codes.count(char_1) > 0:
            char_1_deabbreviate_costs = deabbreviate_costs[char_1]

        for j in range(1, n + 1):
            char_2_code = string_2_char_codes[j - 1]

            substitute_cost = substitute_costs[char_1_code, char_2_code]
            delete_cost = delete_costs[char_1_code]
            insert_cost = insert_costs[char_2_code]

            min_cost = dp[i - 1, j] + delete_cost
            if dp[i, j - 1] + insert_cost < min_cost:
                min_cost = dp[i, j - 1] + insert_cost
            if dp[i - 1, j - 1] + substitute_cost < min_cost:
                min_cost = dp[i - 1, j - 1] + substitute_cost
            
            if deabbreviate_char_codes.count(char_1) == 0:
                dp[i, j] = min_cost
                continue

            for k in range(1, min(j, deabbreviate_max_len) + 1):
                span_cost = 0
                for l in range(k):
                    span_cost += char_1_deabbreviate_costs[
                        string_2_char_codes[j + l - k]
                    ]
                span_cost = span_cost * pow(deabbreviate_penalty, k)
                deabbreviate_cost = dp[i - 1, j - k] + span_cost
                if deabbreviate_cost < min_cost:
                    min_cost = deabbreviate_cost

            dp[i, j] = min_cost

    return dp[m, n]

def lde_similarity(
    str string_1,
    str string_2,
    cnp.ndarray[double, ndim=2] substitute_costs,
    cnp.ndarray[double, ndim=1] insert_costs,
    cnp.ndarray[double, ndim=1] delete_costs,
    dict deabbreviate_costs,
    double deabbreviate_penalty = 1.0,
    int deabbreviate_max_len = 0
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
        deabbreviate_penalty,
        deabbreviate_max_len
    )

    cdef int len_1 = len(string_1)
    cdef int len_2 = len(string_2)
    if len_1 >= len_2:
        max_len = len_1
    else:
        max_len = len_2

    return 1 - (edit_distance / max_len)
