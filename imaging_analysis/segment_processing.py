#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
segment_processing.py: Python script that contains functions for segment
processing.
"""


__author__ = "DM Brady"
__datewritten__ = "06 Mar 2018"
__lastmodified__ = "22 Mar 2018"


import quantities as pq
import neo
import pandas as pd

def TruncateSegment(segment, start=0, end=0, clip_same=True, evt_start=None, evt_end=None):
    """Given a Segment object, will remove the first 'start' seconds and the 
    last 'end' seconds for all events and analog signals in that segment. If not
    clip_same, specify evt_start and evt_end if you want events to be trimmed
    differently."""
    from imaging_analysis.event_processing import TruncateEvents
    from imaging_analysis.signal_processing import TruncateSignals
    # Makes sure signal is an AnalogSignal object
    if not isinstance(segment, neo.core.Segment):
        raise TypeError('%s must be a Segment object.' % segment)
    # Truncate signals first
    segment.analogsignals = TruncateSignals(segment.analogsignals, 
        start=start, end=end)
    # if clip_same=True, it clips events before/after newly trimmed 
    # analog signal, otherwise you can specify.
    if clip_same:
        segment.events = TruncateEvents(segment.events, 
            start=segment.analogsignals[0].t_start.magnitude, 
            end=segment.analogsignals[0].t_stop.magnitude)
    else:
        segment.events = TruncateEvents(segment.events, start=evt_start, end=evt_end)

    return segment

def TruncateSegments(list_of_segments, start=0, end=0, clip_same=True, evt_start=None, evt_end=None):
    """Given a list of Segment object, will run TruncateSegment on each one."""
    # Make sure list is passed
    if not isinstance(list_of_segments, list):
        raise TypeError('%s must be a list of segment objects' % list_of_segments)
    # Run TruncateSegments
    truncated_list = [TruncateSegment(x, start=start, end=end, clip_same=clip_same, 
        evt_start=evt_start, evt_end=evt_end) for x in list_of_segments]
    return truncated_list

def AppendDataframesToSegment(segment, dataframe):
    """Given a segment object, will check to see if it has the correct attribute.
    If not, will create 'dataframes' attribute and append dataframe to it or cycle
    through a list of dataframes and append them all."""
    # Checks segment and dataframe types
    if not isinstance(segment, neo.core.Segment):
        raise TypeError('%s must be a segment object' % segment)
    if isinstance(dataframe, pd.core.frame.DataFrame):
        dataframe = [dataframe]
    elif isinstance(dataframe, list) and \
        all(isinstance(x, pd.core.frame.DataFrame) for x in dataframe):
        dataframe = dataframe 
    else:
        raise TypeError('%s must be a dataframe or list of dataframes object' % dataframe)
    # checks if attribute exists, if not creates it
    if not hasattr(segment, 'dataframes'):
        segment.dataframes = []
    # Adds dataframe to segment object
    for df in dataframe:
        segment.dataframes.append(df)

def AlignEventsAndSignals(seg=None, epoch_name=None, analog_ch_name=None, 
        event_ch_name=None, event=None, event_type='type', 
        prewindow=0, postwindow=0, window_type='event', clip=False):
    """Takes a segment object and spits out four dataframes:
    1) all analog traces for a specified epoch and window centered at an event
    2) all event traces for a specified epoch and window centered at an event
    2) average analog trace +/- sem for a specified epoch and window
    3) average analog value of the average analog trace +/- sem for a 
    specified epoch and window

    Takes the following arguments:
    1) seg: segment object to be processed. Must have processed analog signals,
    events, epochs, and dataframes
    2) epoch: the type of epoch to analyze (e.g. 'omission', 'correct_correct')
    3) event: the event to align by (e.g. 'result', 'iti_start', 'start')
    4) event_type: 'label' means the exact name of an event ('iti_start',
    'tray_activated'), 'type' means the type of event specified in event_params.json
    ('results', 'start', 'end')
    5) prewindow: time in seconds to begin collecting before event or trial
    6) postwindow: time in seconds to end collecting after event or trial
    7) window_type: 'event' means the pre/post windows are only around the alignment
    event. 'trial' means the pre/post windows are collected around the entire trial
    8) clip: only relevant for window_type='trial', clip=True will only keep values
    within the specified window (even though trials may be different lengths), so
    there will be NaN values for shorter trials. clip=False will use the longest
    trial as a default and just collect the values. clip=False is more relevant
    for plotting.

    Example:
    AlignEventsAndSignals(segment, epoch_name='correct_incorrect', 
        analog_ch_name='deltaf_f', event_ch_name='Events', event='start', 
        event_type='type', prewindow=2, postwindow=8, window_type='event')

    Will find all 'correct_incorrect' trials, look for the first event type called
    'start' in each trial, and collect 2 seconds before and 8 seconds after that
    event.

    Example 2:
    AlignEventsAndSignals(segment, epoch_name='incorrect', analog_ch_name='deltaf_f',
        event_ch_name='Events', event='tray_activated', event_type='label',
        prewindow=1, postwindow=1, window_type='trial', clip=True)

    Will find all 'incorrect' trials, and take 1 seconds before the start of each
    trial to 1 second after each trial. The data will be aligned at tray_activated.
    Since clip=True, the dataframe will be as long as the longest trial, and so shorter
    trials will have NaN values outside of their pre/post window.
    """
    # Check segment object is passed
    if not isinstance(seg, neo.core.Segment):
        raise TypeError('seg variable must be a segment object')

    # Extract epoch object
    try:
        epoch = filter(lambda x: x.name == epoch_name, seg.epochs)[0]
    except:
        raise ValueError("""%s not in segment object. Did you not run 
            GroupTrialsByEpoch or misspell the epoch_name?"""
            % epoch_name)

    # Extract analog signal object
    try:
        signal = filter(lambda x: x.name == analog_ch_name, seg.analogsignals)[0]
    except:
        raise ValueError("""%s not in segment object. Did you not run 
            ProcessSignalData or misspell the analog_ch_name?"""
            % analog_ch_name)

