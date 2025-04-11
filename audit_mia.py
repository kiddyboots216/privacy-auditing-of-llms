import os
import logging

import time
import sys
import sklearn
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import numpy as np
import pickle
import math
import copy
import string
import scipy

def audit(scores, y_true, num_secrets, conf=0.01, delta=1e-5):
    # sorted_indices, y_true, num_secrets = extract(path)
    shuffle_ids = np.arange(len(scores))
    np.random.shuffle(shuffle_ids)
    scores = scores[shuffle_ids]
    y_true = y_true[shuffle_ids]
    sorted_indices = np.argsort(scores)
    # print(sorted_indices)
    # import pdb; pdb.set_trace()
    max_est_eps = 0
    num_guesses_for_max = 0
    for num_guesses in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]: #range(1, 20):
        top_guesses = sorted_indices[:num_guesses]
        correct_guesses = y_true[top_guesses].sum()
        eps_est = get_eps_audit(num_secrets, int(num_guesses), int(correct_guesses), delta, conf)
        if eps_est > max_est_eps:
            max_est_eps = eps_est
            num_guesses_for_max = num_guesses
    #if max_est_eps > 0.1:
    print(f"{max_est_eps:.4f} at {num_guesses_for_max} guesses with p value = {conf}.")
    
    return max_est_eps #arr.append((max_est_eps, path))

def p_value_DP_audit(m, r, v, eps, delta):
    if r == 0 or v == 0:
        return 0
    assert 0 <= v <= r <= m
    assert eps >= 0
    assert 0 <= delta <= 1
    q = 1/(1+math.exp(-eps)) # accuracy of eps-DP randomized response
    beta = scipy.stats.binom.sf(v-1, r, q) # = P[Binomial(r, q) >= v]
    alpha = 0
    sum = 0 # = P[v > Binomial(r, q) >= v - i]
    for i in range(1, v + 1):
        sum = sum + scipy.stats.binom.pmf(v - i, r, q)
        if sum > i * alpha:
            alpha = sum / i
        p = beta + alpha * delta * 2 * m
    return min(p, 1)
    # m = number of examples, each included independently with probability 0.5
    # r = number of guesses (i.e. excluding abstentions)
    # v = number of correct guesses by auditor
    # p = 1-confidence e.g. p=0.05 corresponds to 95%
    # output: lower bound on eps i.e. algorithm is not (eps,delta)-DP
def get_eps_audit(m, r, v, delta, p):
    if r == 0 or v == 0:
        return 0
    assert 0 <= v <= r <= m
    assert 0 <= delta <= 1
    assert 0 < p < 1
    eps_min = 0 # maintain p_value_DP(eps_min) < p
    eps_max = 1 # maintain p_value_DP(eps_max) >= p
    while p_value_DP_audit(m, r, v, eps_max, delta) < p:
        eps_max = eps_max + 1
    for _ in range(30): # binary search
        eps = (eps_min + eps_max) / 2
        if p_value_DP_audit(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps
    return eps_min
