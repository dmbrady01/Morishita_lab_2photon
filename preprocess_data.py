#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
preprocess_data.py: Python script that processes fiber photometry data. It truncates
the signal, filters it, labels events, processes trials, and groups trials by epoch.
"""


__author__ = "DM Brady"
__datewritten__ = "07 Mar 2018"
__lastmodified__ = "06 Feb 2019"

import sys
from imaging_analysis.event_processing import LoadEventParams, ProcessEvents, ProcessTrials, GroupTrialsByEpoch, GenerateManualEventParamsJson
from imaging_analysis.segment_processing import TruncateSegments, AppendDataframesToSegment
from imaging_analysis.utils import ReadNeoPickledObj, ReadNeoTdt, WriteNeoPickledObj, PrintNoNewLine
from imaging_analysis.signal_processing import ProcessSignalData
import numpy as np
from neo.core import Epoch, Event
import quantities as pq
import pandas as pd
#######################################################################
# VARIABLES TO ALTER
# For truncating the recordings/events
truncate_begin = 10 # how many seconds to remove from the beginning of the recording
truncate_end = 0 # how many seconds to remove from the end of the recording
# For filtering and calculating deltaf/f
lowpass_filter = 40.0 # Will remove frequencies above this number
signal_channel = '465A 1' # Name of our signal channel
reference_channel = '405A 1' # Name of our reference channel
#deltaf_ch_name = 'DeltaF_F' # New name for our processed signal channel
deltaf_options = {} # Any parameters you want to pass when calculating deltaf/f
#deltaf_options = {'mode': 'z_score', 'subtract': False} # Any parameters you want to pass when calculating deltaf/f
# For calculating events and event labels
tolerance = .1 # Tolerance window (in seconds) for finding coincident events
#processed_event_ch_name = 'Events'
# How is a trial considered over? The 'last' event in a trial or the first event
# in the 'next' trial?
how_trial_ends = 'last'
# Save processed trials as pickle object? Honestly, its faster just to run 
# the processing again
load_pickle_object = False
save_pickle = False 
pickle_name = 'processed.pkl'

####### What mode is the programming running? If TTL, then ProcessEvents is run
# Otherwise you need to add your events manually
mode = 'TTL'
path_to_ttl_event_params = 'imaging_analysis/ttl_event_params_new_rig.json'
print('\n\n\n\nRUNNING IN MODE: %s \n\n\n' % mode)
path_to_social_excel = '/Users/DB/Development/Monkey_frog/data/social/FP_41718_PVGHjSI_9949_2_social.csv'
##########################################################################

# Checks if a directory path to the data is provided, if not, will
# use what is specified in except
try:
    dpath = sys.argv[1]
except IndexError:
    #dpath = '/Users/DB/Development/Monkey_frog/data/social/TDT-LockinRX8-22Oct2014_20-4-15_DT1_041718'
    dpath = '/Users/DB/Development/Monkey_frog/data/KN_newRigData/RS/12/FirstFibPho-180817-160254/'

# Tries to load a processed pickle object, othewise reads the Tdt folder,
# processes the data and writes a pickle object
try:
    # Attempting to load pickled object
    if load_pickle_object:
        PrintNoNewLine('Trying to load processed pickled object...')
        seglist = ReadNeoPickledObj(path=dpath, name=pickle_name, 
                                return_block=False)
        print('Done!')
    else:
        raise IOError
except IOError:
    # Reads data from Tdt folder
    PrintNoNewLine('\nCannot find processed pkl object, reading TDT folder instead...')
    block = ReadNeoTdt(path=dpath, return_block=True)
    seglist = block.segments
    print('Done!')

    # Trunactes first/last seconds of recording
    PrintNoNewLine('Truncating signals and events...')
    seglist = TruncateSegments(seglist, start=truncate_begin, end=truncate_end, clip_same=True)
    print('Done!')

    # Iterates through each segment in seglist. Right now, there is only one segment
    for segment in seglist:
        # Extracts the sampling rate from the signal channel
        try:
            sampling_rate = filter(lambda x: x.name == signal_channel, segment.analogsignals)[0].sampling_rate
        except IndexError:
            raise ValueError('Could not find your channels. Make sure you have the right names!')
        # Appends an analog signal object that is delta F/F. The name of the channel is
        # specified by deltaf_ch_name above. It is calculated using the function
        # NormalizeSignal in signal_processing.py. As of right now it:
        # 1) Lowpass filters signal and reference (default cutoff = 40 Hz, order = 5)
        # 2) Calculates deltaf/f for signal and reference (default is f - median(f) / median(f))
        # 3) Detrends deltaf/f using a savgol filter (default window_lenght = 3001, poly order = 1)
        # 4) Subtracts reference from signal
        # NormalizeSignal has a ton of options, you can pass in paramters using
        # the deltaf_options dictionary above. For example, if you want it to be mean centered
        # and not run the savgol_filter, set deltaf_options = {'mode': 'mean', 'detrend': False}
        PrintNoNewLine('\nCalculating delta_f/f...')
        ProcessSignalData(seg=segment, sig_ch=signal_channel, ref_ch=reference_channel,
                        name='DeltaF_F', fs=sampling_rate, highcut=lowpass_filter, **deltaf_options)
        # Appends an Event object that has all event timestamps and the proper label
        # (determined by the evtframe loaded earlier). Uses a tolerance (in seconds)
        # to determine if events co-occur. For example, if tolerance is 1 second
        # and ch1 fires an event, ch2 fires an event 0.5 seconds later, and ch3 fires
        # an event 3 seconds later, the output array will be [1, 1, 0] and will
        # match the label in evtframe (e.g. 'omission')
        print('Done!')
        if mode == 'TTL':
            # Loading event labeling/combo parameters
            path_to_event_params = path_to_ttl_event_params
        elif mode == 'manual':
            # Generates a json for reading excel file events
            path_to_event_params = 'imaging_analysis/manual_event_params.json'
            GenerateManualEventParamsJson(path_to_social_excel, event_col='Bout type', 
                name=path_to_event_params)
        # This loads our event params json
        start, end, epochs, evtframe, typeframe = LoadEventParams(dpath=path_to_event_params, 
            mode=mode)
        # Appends processed event_param.json info to segment object
        AppendDataframesToSegment(segment, [evtframe, typeframe], 
            ['eventframe', 'resultsframe'])
        # Processing events
        PrintNoNewLine('\nProcessing event times and labels...')
        ProcessEvents(seg=segment, tolerance=tolerance, evtframe=evtframe, 
            name='Events', mode=mode, manualframe=path_to_social_excel, 
            event_col='Bout type', start_col='Bout start', end_col='Bout end')
        print('Done!')
        # Takes processed events and segments them by trial number. Trial start
        # is determined by events in the list 'start' from LoadEventParams. This
        # can be set in the event_params.json. Additionally, the result of the 
        # trial is set by matching the epoch type to the typeframe dataframe 
        # (also from LoadEventParams). Example of epochs are 'correct', 'omission',
        # etc. 
        # The result of this process is a dataframe with each event and their
        # timestamp in chronological order, with the trial number and trial outcome
        # appended to each event/timestamp.
        PrintNoNewLine('\nProcessing trials...')
        trials = ProcessTrials(seg=segment, name='Events', 
            startoftrial=start, epochs=epochs, typedf=typeframe, 
            appendmultiple=False)
        print('Done!')
        # With processed trials, we comb through each epoch ('correct', 'omission'
        # etc.) and find start/end times for each trial. Start time is determined
        # by the earliest 'start' event in a trial. Stop time is determined by
        # 1) the earliest 'end' event in a trial, 2) or the 'last' event in a trial
        # or the 3) 'next' event in the following trial.
        PrintNoNewLine('\nCalculating epoch times and durations...')
        GroupTrialsByEpoch(seg=segment, startoftrial=start, endoftrial=end, 
            endeventmissing=how_trial_ends)
        print('Done!')
        # elif mode == 'manual':
        #     trials = pd.DataFrame({'time': 90.0, 'event': 'exposure', 
        #         'results': 'exposure', 'with_previous_results': np.nan, 
        #         'event_type': 'start', 'trial_idx': 1}, index=range(1))
        #     AppendDataframesToSegment(segment, [trials], ['trials'])
        #     segment.epochs.append(Epoch(times=np.array([90.0]) * pq.s, 
        #         durations=np.array([0]) * pq.s, name='exposure'))
        #     segment.events.append(Event(times=np.array([90.0]) * pq.s, 
        #         labels=np.array(['exposure'], dtype='S'), name='Events'))
        # # add processed flag to segment
        segment.processed = True
    # Option to save pickle object
    if save_pickle:
        # Saves everything to pickeled object
        PrintNoNewLine('\nWriting processed pickled object...')
        WriteNeoPickledObj(block, path=dpath, name=pickle_name)
        print('Done')

