#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
segment_processing.py: Python script that contains functions for segment
processing.
"""


__author__ = "DM Brady"
__datewritten__ = "06 Mar 2018"
__lastmodified__ = "16 Mar 2018"


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
