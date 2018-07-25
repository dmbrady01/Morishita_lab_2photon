#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
markov.py: Python script that contains functions for making markov models.
"""


__author__ = "DM Brady"
__datewritten__ = "19 Jul 2018"
__lastmodified__ = "25 Jul 2018"


# Required modules
import pandas as pd
import numpy as np
import os
from imaging_analysis.event_processing import FormatManualExcelFile

def ReadStateCsv(state_csv='markov/states.csv'):
    "Reads the possible state csv"
    return pd.read_csv(state_csv)

def GetTransitionsFromExcel(excel_file=None, column='Bout type'):
    "Reads the excel file to be processed and returns the Series we care about"
    return FormatManualExcelFile(excel_file=excel_file, event_col=column)[column]

# def AddStartAndEnds(df):
#     "Adds starts and ends to Series with transitions"
#     return pd.concat([pd.Series(['start']), df, pd.Series(['end'])]).reset_index(drop=True)

def StateMapping(df, state_csv='markov/states.csv'):
    "Reads the current transition sequence to be analyzed and updates states_csv. Returns numerical codes."
    set_df = set(df.unique())
    # Read state csv
    state_code = ReadStateCsv(state_csv=state_csv)
    set_state = set(state_code['states'].unique())
    # See if states in set_df are not in set_state
    not_mapped = set_df.difference(set_state)
    if len(not_mapped) > 0:
        state_code = state_code.append(pd.DataFrame(list(not_mapped), columns=['states']), ignore_index=True)
        state_code.to_csv('markov/states.csv', index=False)

    return {y:x for x, y in state_code['states'].to_dict().iteritems()}

def EncodeStates(df, code):
    "Given a dataframe and a state_code, creates a list of transitions"
    return df.map(code).values

def CountMatrix(transitions):
    "Given a list of transitions and the code, creates a count matrix"
    num_states = 1 + max(transitions)

    matrix = [[0]*num_states for _ in range(num_states)]

    for (i,j) in zip(transitions,transitions[1:]):
        matrix[i][j] += 1.0

    return np.array(matrix)

def ProcessExcelToCountMatrix(excel_file, column='Bout type', state_csv='markov/states.csv'):
    "Reads an excel file and states csv, updates states csv, outputs count matrix"
    df = GetTransitionsFromExcel(excel_file=excel_file, column=column)
    code = StateMapping(df, state_csv=state_csv)
    transitions = EncodeStates(df, code)
    count_matrix = CountMatrix(transitions)
    return count_matrix, transitions

def RightStochasticMatrix(count_matrix):
    "Takes a count matrix and generates a stochastic matrix"
    return np.array([np.divide(count_matrix[:,x], count_matrix.sum(axis=1)) 
        for x in range(count_matrix.shape[0])]).T

def AddEmptyRowAndColumn(matrix, goal_shape=10):
    shape = matrix.shape[0]
    num = goal_shape - shape
    if num > 0:
        new_matrix = np.zeros((shape + num, shape + num))
        new_matrix[:-num, :-num] = matrix
        return new_matrix
    else:
        return matrix

def AddingCountMatrices(list_of_count_matrices):
    shapes_list = [x.shape[0] for x in list_of_count_matrices]
    max_shape = max(shapes_list)
    reshaped = [AddEmptyRowAndColumn(x, max_shape) for x in list_of_count_matrices]
    sum_of_matrices = reshaped[0]
    for matrix in reshaped[1:]:
        sum_of_matrices = np.add(sum_of_matrices, matrix)
    return sum_of_matrices