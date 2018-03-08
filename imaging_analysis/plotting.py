#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plotting.py: Python script that contains functions for plotting data.
"""


__author__ = "DM Brady"
__datewritten__ = "07 Mar 2018"
__lastmodified__ = "07 Mar 2018"

import matplotlib.pyplot as plt
import seaborn as sns

def PlotSignalProcessing(signal_list=None, time=None, ylabels=None, titles=None):
    """Given a list (or even list of lists if you want to plot several together), 
    creates a single figure for looking at Signal data."""
    if not isinstance(signal_list, list):
        raise TypeError('%s must be a list' % signal_list)
    # Sets background style
    sns.set_style('whitegrid')
    # Calculates number of plots
    num_plots = len(signal_list) 
    # Prepare subplot indexing
    index = num_plots * 100 + 11
    # Starts figure
    fig = plt.figure()
    for x in range(num_plots):
        # sets up first graph
        if x == 0:
            original_axis = fig.add_subplot(index + x)
            plt.plot(time)

        axis = index + x
