#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
markov_permutation_test.py: Python script for performing permutation test on two sets of transition.txts.
"""


__author__ = "DM Brady"
__datewritten__ = "31 Jul 2018"
__lastmodified__ = "31 Jul 2018"

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from markov.markov import ReadTransitionsTextFile, MaxStates, MarkovToTransitionMatrix, StationaryDistribution, DistanceBewtweenMatrices


# Paths to transition.txt files
group1 = '/Users/DB/Development/Monkey_frog/data/social/csvs/group_housed_transistions.txt'
group2 = '/Users/DB/Development/Monkey_frog/data/social/csvs/isolates_transistions.txt'

# Number of permutations to run
num_permutations = 10000
np.random.seed(1234) # sets the random seed generator to get the same distro each time

# Read data into list of lists
group1_data = ReadTransitionsTextFile(group1)
group2_data = ReadTransitionsTextFile(group2)

# Get max number of states
num_states = MaxStates(group1_data, group2_data)

# Real differences (finds the stationary distro for each)
group1_sd = StationaryDistribution(MarkovToTransitionMatrix(group1_data, num_states=num_states, replace_nan=True))
group2_sd = StationaryDistribution(MarkovToTransitionMatrix(group2_data, num_states=num_states, replace_nan=True))
true_distance = DistanceBewtweenMatrices(group1_sd, group2_sd)

# Preparing for permutation test
group1_sample_size = len(group1_data)
group2_sample_size = len(group2_data)
total_n = group1_sample_size + group2_sample_size
combined_data = np.array(group1_data + group2_data)
null_distribution = []

while len(null_distribution) < num_permutations:
    # random permutation
    shuffle = np.random.permutation(total_n).astype(int)
    group1_idx = shuffle[:group1_sample_size]
    group2_idx = shuffle[group1_sample_size:]

    # Shuffled data
    group1_perm = combined_data[group1_idx]
    group2_perm = combined_data[group2_idx]

    # Calcualte stationary distributions
    group1_perm_sd = StationaryDistribution(MarkovToTransitionMatrix(group1_perm, num_states=num_states, replace_nan=True))
    group2_perm_sd = StationaryDistribution(MarkovToTransitionMatrix(group2_perm, num_states=num_states, replace_nan=True))
    perm_distance = DistanceBewtweenMatrices(group1_perm_sd, group2_perm_sd)
    null_distribution.append(perm_distance) 

# For plotting
null_distribution = np.array(null_distribution)
alpha = np.percentile(null_distribution, 95)
p_value = np.sum(null_distribution > true_distance)/float(null_distribution.shape[0])

histplot = sns.distplot(null_distribution, norm_hist=True)
histplot.axvline(alpha, color='r', linestyle='--')
histplot.axvline(true_distance, color='k', linestyle='-')
fig = histplot.get_figure()
fig.savefig('permutation_test.pdf')

