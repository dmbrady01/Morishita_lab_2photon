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
import os

def PlotAverageSignal(traces, mode='raw', events=[0, 5], sem=True, save=True, 
        title=None, color='b', alpha=0.1, dpath=''):
    """Given a dataframe of traces (mode='raw') or an average trace (mode='avg'). 
    It will draw the average trace +/- the sem (if sem=True) or the sd (sem=False).
    events is a list of times when there should be vertical lines (trial start, 
    stimulus onset, end, etc.)"""
    if mode == 'raw':
        avg = traces.mean(axis=1)
        if sem:
            error = traces.sem(axis=1)
        else:
            error = traces.sd(axis=1)
    elif mode == 'avg':
        avg = traces['avg']
        if sem:
            error = traces['se']
        else:
            error = traces['sd']

    plt.plot(traces.index, avg, color=color)
    plt.fill_between(traces.index, avg-error, avg+error, color=color, alpha=alpha)
    
    if len(events) > 0:
        [plt.axvline(event, color='k', linestyle='--') for event in events]
    
    if title is not None:
        plt.title(title)

    if save:
        if len(dpath) == 0:
            dpath = os.getcwd()
        dpath = dpath + os.sep + title
        plt.savefig(dpath + '.pdf')