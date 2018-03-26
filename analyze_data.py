#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze_data.py: Python script that analyzes fiber photometry data. Clips traces
to be centered around specified events. Outputs csvs and appends them to segment objects.
"""


__author__ = "DM Brady"
__datewritten__ = "23 Mar 2018"
__lastmodified__ = "23 Mar 2018"

import sys
from imaging_analysis.utils import ReadNeoPickledObj, WriteNeoPickledObj, PrintNoNewLine
from imaging_analysis.segment_processing import AlignEventsAndSignals
#######################################################################
# VARIABLES TO ALTER
# for loading your processed data
pickle_name = 'processed.pkl'
load_pickle_object = False
# what epoch do you want to analyze? 'correct', 'correct_correct', etc.
epoch_name = 'correct'
# what is the name of the analog channel to analyze? make sure it is the same
# as in process_data
analog_ch_name = 'DeltaF_F'
# what is the name of the event channel to analyze? make sure it is the same as
# in process_data
event_ch_name = 'Events'
# what is the event to align the trials by?
event = 'iti_start'
# is event above a specific event like 'tray_activated'? if so event_type = 'label'
# or is it a class of events like 'start'? if so event_type = 'type'
event_type = 'type'
# what is the pre window (in seconds)
prewindow = 2
# what is the post window (in seconds)
postwindow = 8
# what type of window do you want? just around the event? ('event') or around the
# entire trial ('trial')? Around the trial is probably used for plotting
window_type = 'event'
# Do you want to clip the signal (only relevant if window_type = 'trial')
# If clipped, then every trial has values only between prewindow and postwindow
# even though trial lengths differ. That means there will be NaN values. If
# not clipped, then the windows are determined by the trials with the longest
# pre and post windows around the trial. You want clip=False for plotting
clip = False
# What do you want to call this analysis? i.e. 2seconds_iti_start_8seconds, etc.
name = 'correct_start'
# Do you want to save to csv? All data is saved in each segments 'analyzed'
# attribute
to_csv = True
# PLOTTING parameters
save_plot = True 
plot_title = 'Correct Trials'


try:
    dpath = sys.argv[1]
except IndexError:
    dpath = '/Users/DB/Development/Monkey_frog/data/TDT-LockinRX8-22Oct2014_20-4-15_DT4_1024174'

# Tries to load a processed pickle object, othewise tells you ro run process_data first
if load_pickle_object:
    try:
        # Attempting to load pickled object
        PrintNoNewLine('Trying to load processed pickled object...')
        seglist = ReadNeoPickledObj(path=dpath, name=pickle_name, 
                                    return_block=False)
        print('Done!')
    except:
        raise OSError('You do not have a pickled object. Did you run process_data first?')

for segment in seglist:
    PrintNoNewLine('Centering trials and analyzing...')
    AlignEventsAndSignals(seg=segment, epoch_name=epoch_name, analog_ch_name=analog_ch_name, 
        event_ch_name=event_ch_name, event=event, event_type=event_type, 
        prewindow=prewindow, postwindow=postwindow, window_type=window_type, 
        clip=clip, name=name, to_csv=to_csv, dpath=dpath)
    print('Done!')