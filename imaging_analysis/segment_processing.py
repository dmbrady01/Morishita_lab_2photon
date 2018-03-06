#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
segment_processing.py: Python script that contains functions for segment
processing.
"""


__author__ = "DM Brady"
__datewritten__ = "06 Mar 2018"
__lastmodified__ = "06 Mar 2018"


import quantities as pq
import neo
from imaging_analysis.event_processing import TruncateEvents
from imaging_analysis.signal_processing import TruncateSignals

def TruncateSegment(segment, start=0, end=0, clip_same=True, evt_start=None, evt_end=None):
    """Given a Segment object, will remove the first 'start' seconds and the 
    last 'end' seconds for all events and analog signals in that segment. If not
    clip_same, specify evt_start and evt_end if you want events to be trimmed
    differently."""
    # Makes sure signal is an AnalogSignal object
    if not isinstance(segment, neo.core.Segment):
        raise TypeError('%s must be a Segment object.' % segment)
    # Truncate signals first
    segment.analogsignals = TruncateSignals(segment.analogsignals, \
                                            start=start, end=end)
    # if clip_same=True, it clips events before/after newly trimmed 
    # analog signal, otherwise you can specify.
    if clip_same:
        segment.events = TruncateEvents(segment.events, \
                                    start=segment.analogsignals[0].t_start.magnitude, \
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
    truncated_list = [TruncateSegment(x, start=start, end=end, clip_same=clip_same, \
                        evt_start=evt_start, evt_end=evt_end) for x in list_of_segments]
    return truncated_list