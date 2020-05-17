from vi_and_pi import policy_evaluation, policy_improvement
import numpy as np


def test_policy_evaluation():
    P = {
        0: {0: [(1.0, 1, 5, False)]},
        1: {1: [(1.0, 1, -1, False)]}
     }
    nS = 2
    nA = 2
    policy = [0, 1]
    tol = 2.0
    gamma = 1.0

    res = np.array([4, -2])
    np.testing.assert_array_equal(res, policy_evaluation(P, nS, nA, policy, gamma=gamma, tol=tol))


def test_policy_improvement():
    P = {
        0: {0: [(1.0, 1, 5, False)], 1: [(0.0, 1, 5, False)]},
        1: {0: [(0.0, 1, -1, False)], 1: [(1.0, 1, -1, False)]}
     }
    nS = 2
    nA = 2
    policy = [0, 1]
    tol = 2.0
    gamma = 1.0
    value_from_policy = np.array([4, -2])
    new_policy = np.array([0, 0])

    np.testing.assert_array_equal(new_policy, policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9))
