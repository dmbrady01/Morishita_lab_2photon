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

dpath = '/Users/DB/Development/Monkey_frog/data/TDT-LockinRX8-22Oct2014_20-4-15_DT4_1024175'
name = 'new_data'


# CHANGE HERE (all_traces for single animal, average_trace for mult-animal)
csv1 = '/Users/DB/Development/Monkey_frog/data/TDT-LockinRX8-22Oct2014_20-4-15_DT4_1024174/all_traces.csv'
csv2 = '/Users/DB/Development/Monkey_frog/data/TDT-LockinRX8-22Oct2014_20-4-15_DT4_1024175/all_traces.csv'
csvs = [csv1, csv2] # add csv3, cs4 here
multi_animal = True # True if combining multiple animals. Make sure csv is average_trace!

# For plotting
plot_events = [0, 5]
sem = True
save_plot = True
plot_title = 'My plot title'
color = 'b'
alpha = 0.1

combined_list = []
for csv in csvs:
    data1 = pd.read_csv(csv)
    if multi_animal is True:
        data1 = data1.avg
    combined_list.append(data1)

combined = pd.concat(combined_list, axis=1)

# Calculate average signal
# Calculate average signal
avg_df = pd.DataFrame()
avg_df['avg'] = combined.mean(axis=1)
avg_df['sd'] = combined.std(axis=1)
avg_df['se'] = combined.sem(axis=1)

# Calculate average of average (point estimate)
pe_df = pd.DataFrame(np.nan, columns=['avg', 'sd', 'se'], index=[0])
pe_df.loc[0, 'avg'] = avg_df.avg.mean()
pe_df.loc[0, 'sd'] = avg_df.sd.mean()
pe_df.loc[0, 'se'] = avg_df.se.sem()

dpath = dpath + os.sep + name
combined.to_csv(dpath + '_all_traces.csv')
avg_df.to_csv(dpath + '_average_trace.csv')
pe_df.to_csv(dpath + '_point_estimate.csv')

PrintNoNewLine('Plotting data...')
PlotAverageSignal(combined, mode='raw', events=plot_events, sem=sem, save=save_plot, 
    title=plot_title, color=color, alpha=alpha, dpath=dpath)
print('Done!')

