#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze_data.py: Python script that analyzes fiber photometry data. Clips traces
to be centered around specified events. Outputs csvs and appends them to segment objects.
"""


__author__ = "DM Brady"
__datewritten__ = "23 Mar 2018"
__lastmodified__ = "23 Mar 2018"

import pandas as pd
import os
import numpy as np
from imaging_analysis.utils import PrintNoNewLine
from imaging_analysis.plotting import PlotAverageSignal

dpath = '/Users/DB/Development/Monkey_frog/data/'
name = 'three_days_from_one_animal'


# CHANGE HERE (all_traces for single animal, average_trace for mult-animal)
csv1 = '/Users/DB/Development/Monkey_frog/data/TDT-LockinRX8-22Oct2014_20-4-15_DT4_1024174/correct_stimulus_all_traces.csv'
csv2 = '/Users/DB/Development/Monkey_frog/data/TDT-LockinRX8-22Oct2014_20-4-15_DT4_1024175/correct_stimulus_all_traces.csv'
csv3 = '/Users/DB/Development/Monkey_frog/data/TDT-LockinRX8-22Oct2014_20-4-15_DT4_1024173/correct_stimulus_all_traces.csv'
csvs = [csv1, csv2, csv3] # add csv3, cs4 here

# If all_traces.csv is chosen, it is assumed to be combining info from one animal
if 'all_traces' in np.random.choice(csvs):
    multi_animal = False 
else:
    multi_animal = True # True if combining multiple animals. Make sure csv is average_trace!

# For plotting
plot_events = [0, 5]
sem = True
save_plot = True
plot_title = 'Three animals combined'
color = 'b'
alpha = 0.1

# Set these to numbers if you want to control axis size
xmin = None
xmax = None 
ymin = None 
ymax = None

# smoothing window in msecs (None if you don't want to use it.)
smoothing_window = None

combined_list = []
for csv in csvs:
    data1 = pd.read_csv(csv, index_col=0)
    if multi_animal is True:
        data1 = data1.loc[:, 'avg']
    combined_list.append(data1)

combined = pd.concat(combined_list, axis=1)
combined = combined.bfill().ffill()

# calculate sampling_rate
sampling_rate = 1./(combined.index[-1] - combined.index[-2])
# Calculate average signal
# Calculate average signal
avg_df = pd.DataFrame(index=combined.index)
avg_df['avg'] = combined.mean(axis=1)
avg_df['sd'] = combined.std(axis=1)
avg_df['se'] = combined.sem(axis=1)

# Calculate average of average (point estimate)
pe_df = pd.DataFrame(np.nan, columns=['avg', 'sd', 'se'], index=[0])
pe_df.loc[0, 'avg'] = avg_df.avg.mean()
pe_df.loc[0, 'sd'] = avg_df.sd.mean()
pe_df.loc[0, 'se'] = avg_df.se.sem()

csv_dpath = dpath + os.sep + name
combined.to_csv(csv_dpath + '_all_traces.csv')
avg_df.to_csv(csv_dpath + '_average_trace.csv')
pe_df.to_csv(csv_dpath + '_point_estimate.csv')

PrintNoNewLine('Plotting data...')
PlotAverageSignal(combined, mode='raw', events=plot_events, sem=sem, save=save_plot, 
    title=plot_title, color=color, alpha=alpha, dpath=dpath, xmin=xmin, xmax=xmax,
    ymin=ymin, ymax=ymax, smoothing_window=smoothing_window, sampling_frequency=sampling_rate)
print('Done!')

