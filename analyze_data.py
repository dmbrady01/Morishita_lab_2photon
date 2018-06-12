#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze_data.py: Python script that analyzes fiber photometry data. Clips traces
to be centered around specified events. Outputs csvs and appends them to segment objects.
"""


__author__ = "DM Brady"
__datewritten__ = "23 Mar 2018"
__lastmodified__ = "11 Jun 2018"

import sys
from imaging_analysis.utils import ReadNeoPickledObj, WriteNeoPickledObj, PrintNoNewLine
from imaging_analysis.segment_processing import AlignEventsAndSignals
from imaging_analysis.plotting import PlotAverageSignal
#######################################################################
# VARIABLES TO ALTER
# for loading your processed data
pickle_name = 'processed.pkl'
load_pickle_object = False

analysis_blocks = [
    {
        'epoch_name': 'correct',
        'event': 'iti_start',
        'save_file_as': 'Correct_trials',
        'prewindow': 10,
        'postwindow': 30,
        'window_type': 'event',
        'clip': False,
        'to_csv': True,
        'plot_parameters': {
            'save_plot': True,
            'color': 'b',
            'alpha': 0.1,
            'plot_events': [0],
            'sem': True,
            'xmin': None,
            'xmax': None,
            'ymin': None,
            'ymax': None,
            'smoothing_window': 500
        }
    },
    {
        'epoch_name': 'omission',
        'event': 'omission',
        'save_file_as': 'Omission_trials',
        'prewindow': 5,
        'postwindow': 2,
        'window_type': 'event',
        'clip': False,
        'to_csv': True,
        'plot_parameters': {
            'save_plot': True,
            'color': 'b',
            'alpha': 0.1,
            'plot_events': [0],
            'sem': True,
            'xmin': None,
            'xmax': None,
            'ymin': None,
            'ymax': None,
            'smoothing_window': 500
        }
    }
]
"""
epoch_name: what epoch do you want to analyze

event: what is the event to align by

prewindow: in seconds

postwindow: in seconds

window_type: is the window around the 'event' or entire 'trial'
# Do you want to clip the signal (only relevant if window_type = 'trial')
# If clipped, then every trial has values only between prewindow and postwindow
# even though trial lengths differ. That means there will be NaN values. If
# not clipped, then the windows are determined by the trials with the longest
# pre and post windows around the trial. You want clip=False for plotting

clip: False or True

save_file_as: what do you want to save the file/plot name

to_csv: True/False for writing data to csv

# PLOTTING parameters
save_plot: True/False to save plot
color: color of analog signal line
alpha: opacity of analog signal line
plot_events: times when you want a vertical line in the plot (trial start, stimulus on, etc.)
sem: plot standard error of the mean. if false, plots standard deviation
# Set these to numbers if you want to control axis size
xmin = None
xmax = None
ymin = None 
ymax = None
smoothing_window: smoothing window in msecs (None if you don't want to use it.)
"""

try:
    dpath = sys.argv[1]
except IndexError:
    #dpath = '/Users/DB/Development/Monkey_frog/data/social/TDT-LockinRX8-22Oct2014_20-4-15_DT1_041718'
    dpath = '/Users/DB/Development/Monkey_frog/data/TDT-LockinRX8-22Oct2014_20-4-15_DT4_1024173/'

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
    for block in analysis_blocks:
        # Extract analysis block params
        epoch_name = block['epoch_name']
        event = block['event']
        prewindow = block['prewindow']
        postwindow = block['postwindow']
        window_type = block['window_type']
        clip = block['clip']
        save_file_as = block['save_file_as']
        to_csv = block['to_csv']

        print('\nAnalyzing "%s" trials \n' % epoch_name)

        PrintNoNewLine('Centering trials and analyzing...')
        AlignEventsAndSignals(seg=segment, epoch_name=epoch_name, analog_ch_name='DeltaF_F', 
            event_ch_name='Events', event=event, event_type='label', 
            prewindow=prewindow, postwindow=postwindow, window_type=window_type, 
            clip=clip, name=save_file_as, to_csv=to_csv, dpath=dpath)
        print('Done!')

        traces = segment.analyzed[save_file_as]['all_traces']
        sampling_rate = filter(lambda x: x.name == signal_channel, segment.analogsignals)[0].sampling_rate

        # Extract analysis block plotting params
        plot_params = block['plot_parameters']
        plot_events = plot_params['plot_events']
        sem = plot_params['sem']
        save_plot = plot_params['save_plot']
        color = plot_params['color']
        alpha = plot_params['alpha']
        xmin = plot_params['xmin']
        xmax = plot_params['xmax']
        ymin = plot_params['ymin']
        ymax = plot_params['ymax']
        smoothing_window = plot_params['smoothing_window']
        PrintNoNewLine('Plotting data...')
        PlotAverageSignal(traces, mode='raw', events=plot_events, sem=sem, save=save_plot, 
            title=save_file_as, color=color, alpha=alpha, dpath=dpath, xmin=xmin, xmax=xmax,
            ymin=ymin, ymax=ymax, smoothing_window=smoothing_window, 
            sampling_frequency=sampling_rate)
        print('Done!')