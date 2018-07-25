#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
transition_matrix.py: Python script that contains functions for making markov models.
"""


__author__ = "DM Brady"
__datewritten__ = "19 Jul 2018"
__lastmodified__ = "25 Jul 2018"


import numpy as np
import os
from markov.markov import ProcessExcelToCountMatrix, AddingCountMatrices, RightStochasticMatrix


paths = [
    '/Users/DB/Development/Monkey_frog/data/social/FP_example_object.csv',
    '/Users/DB/Development/Monkey_frog/data/social/FP_example_social.csv'
]
path_to_save = '/Users/DB/Development/Monkey_frog/data/social/'
# Build up your count matrices
count_matrices = []
transitions_list = []
for csv in paths:
    count_matrix, transitions = ProcessExcelToCountMatrix(csv, column='Bout type', state_csv='markov/states.csv')
    count_matrices.append(count_matrix)
    transitions_list.append(list(transitions))


with open(path_to_save + os.sep + "transistions.txt", 'w') as fp:
    for transition in transitions_list:
        fp.write(str(transition) + "\n")

# Add all count matrices and get transistion matric
total_count_matrix = AddingCountMatrices(count_matrices)
transistion_matrix = RightStochasticMatrix(total_count_matrix)
# TO SAVE csv
np.savetxt(path_to_save + os.sep + "count_matrix.csv", total_count_matrix, delimiter=",")
np.savetxt(path_to_save + os.sep + "transistion_matrix.csv", transistion_matrix, delimiter=",")




