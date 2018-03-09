#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
event_processing.py: Python script that contains functions for event and epoch
processing.
"""


__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "09 Mar 2018"

import quantities as pq
import numbers
import neo
from neo.core import Event
import pandas as pd
import numpy as np
import os
import json

def LoadEventParams(dpath=None, evtdict=None):
    """Checks that loaded event parameters (either through a directory path or
    from direct input (dpath vs evtdict)) has the correct structure:
    1) Must contain 'channels' and 'combinations'
    2) Number of channels must equal length of each list for each combination
    Results are returned as a dataframe"""
    # Load event params
    if dpath:
        if not os.path.exists(dpath):
            raise IOError('%s cannot be found. Please check that it exists' % dpath)
        else:
            evtdict = json.load(open(dpath, 'r'))
    else:
        if not isinstance(evtdict, dict):
            raise TypeError('%s must be a dictionary' % evtdict)
    # returns dataframe
    return pd.DataFrame(data=evtdict['combinations'].values(),
        index=evtdict['combinations'].keys(), columns=evtdict['channels'])

def TruncateEvent(event, start=None, end=None):
    """Given an Event object, will remove events before 'start' and after 
    'end'. Start and end must be in seconds. Please note that this is different logic
    than TruncateSignal! TruncateSignals trims start/end seconds from the ends no matter
    the length. TruncateEvent start/end truncates at those specific timestamps. It
    is not relative."""
    # Makes sure event is an Event object
    if not isinstance(event, neo.core.Event):
        raise TypeError('%s must be an Event object.' % event)
    # converts start and end to times
    if start is None:
        start = event.min()
    else:
        start = start * pq.s

    if end is None:
        end = event.max()
    else:
        end = end * pq.s

    truncated_event = event.time_slice(start, end)

    return truncated_event

def TruncateEvents(event_list, start=None, end=None):
    """Given a list of Event objects, will iterate through each one and remove
    events before 'start' and after 'end'. Start and end must be in seconds."""
    # Makes sure a list is passed
    if not isinstance(event_list, list):
        raise TypeError('%s must be a list' % event_list)
    # Iterate through each item with TrunacteEvent
    truncated_list = [TruncateEvent(evt, start=start, end=end) for evt in event_list]
    return truncated_list

def ExtractEventsToList(seg=None, evtframe=None, time_name='times', ch_name='ch'):
    """Given a segment object and an eventframe, extracts event times
    for channels specified in evtframe and returns a list of dictionaries.
    Each dictionary has the event times (time_name) and a number representing
    that channel (ch_name)"""
    # Makes sure segment object is passed
    if not isinstance(seg, neo.core.segment.Segment):
        raise TypeError('%s needs to be a neo segment object' % seg)
    # Makes sure pandas dataframe object is passed
    if not isinstance(evtframe, pd.core.frame.DataFrame):
        raise TypeError('%s needs to be a pandas DataFrame object' % evtframe)
    # Iterates through the event object in seg and pulls out the relevent ones
    eventlist = list()
    for event_obj in seg.events:
        # Checks if event object is type we care about (DIn1, DIn2, etc.)
        if event_obj.name in evtframe.columns:
            # idx to represent channel
            idx = evtframe.columns.get_loc(event_obj.name)
            # Adds dictionary to eventlist containing event times and ch
            eventlist.append({time_name: event_obj.times, ch_name: idx})
    # Makes sure eventlist has something in it
    if len(eventlist) == 0:
        raise ValueError('Segment object does not contain event channels that \
            match those specified in eventframe. Check segment object or \
            event params.json')
    # Makes sure that eventlist length matches evtframe column length
    elif len(eventlist) != evtframe.columns.shape[0]:
        raise ValueError('Evtframe has too many or too few channels compared to \
            the segment.events list. Either segment.events has duplicate entries or \
            event params.json is not correct.')
    else:
        return eventlist

def ProcessEventList(eventlist=None, tolerance=None, evtframe=None, 
    time_name='times', ch_name='ch'):
    """Given an event list of dictionaries (with times in an array associated
    with time_name and channel names associated with ch_name), performs the 
    following algorithm with a while loop:
    1) Copies event dictionary in eventlist to a new list if event dictionary times
    are not empty
    2) Takes the earliest event across the remaining event dictionaries
    3) Creates a temporary list where each element represents whether a channel
    fired at the event time in 2. Default is that the channel did not fire.
    4) Iterates through each channel to see if the earliest event in that channel
    is within the tolerance of the event timestamp from 2).
    5) If that channels event is found to be within that tolerance, it sets that
    channels position in temporary list to 1 (meaning it co-occured).
    6) Events that are determined to co-occur are purged from their respective
    event dictionary.
    7) The co-occurence list is matched against the evtframe to determine the 
    event label.
    8) An event_times list and an event_labels list are updated
    9) When the while loop is finished, event_times and event_labels are returned.
    """
    # Makes sure that tolerance is a number
    if not isinstance(tolerance, numbers.Number):
        raise TypeError('%s needs to be a number' % tolerance)
    # Converts tolerance to have units
    tolerance = tolerance * pq.s
    # Checks that eventlist is a list
    if not isinstance(eventlist, list):
        raise TypeError('%s needs to be a list')
    # Checks time_name and ch_name are keys in each eventlist dict
    time_name_check = all(time_name in d.keys() for d in eventlist)
    ch_name_check = all(ch_name in d.keys() for d in eventlist)
    if not (time_name_check and ch_name_check):
        raise ValueError('%s and %s must be keys in dictionaries in eventlist' 
            % (time_name, ch_name))
    event_times = list() # Will store every event timestamp
    event_labels = list() # Will store every event label
    # Systematically goes through each channel in eventlist. Takes the
    # first event from each channel. Out of those it takes the earliest one.
    # If the other events are within the tolerance of that earliest event,
    # it is marked as co-occuring. All co-occuring events are then deleted from
    # their respective channels. This is done until eventlist event times are
    # empty
    # While loops works until every event is deleted from all channels in
    # event list
    while any(event_array[time_name].size for event_array in eventlist):
        # Pulls out channels that still have events in them
        evtlist_non_empty = filter(lambda x: x[time_name].size, eventlist)
        # Gets the first event for each channel in evtlist_non_empty
        first_elems = map(lambda x: x[time_name][0], evtlist_non_empty)
        # Selects the earliest out of all the channels
        current_earliest = np.amin(first_elems) * pq.s 
        # Sets a list where each element represents whether a channel
        # fired an event or not. Default is that it did not
        current_event_list = np.zeros(evtframe.columns.shape[0])
        # Goes through each channel and sees if the first time stamp 
        # is within the tolerance of the earliest timestamp
        # out of all channels (current_earliest). If so, it sets that channels
        # position in current_event_list to 1 and that timestamp is deleted
        # from eventlist
        for event_ch in evtlist_non_empty:
            # Calculates if event is within tolerance
            if event_ch[time_name][0] - current_earliest <= tolerance:
                # Sets position in current_evet_list to 1
                current_event_list[event_ch[ch_name]] = 1
                # Deletes that event from the channel
                event_ch[time_name] = np.delete(event_ch[time_name], 0) * pq.s
        # Finds the correct label by matching current_event_list (which
        # could look like [0, 1, 1] for example) with the corresponding
        # dataframe row
        label = evtframe.index[evtframe.apply(lambda x: 
            all(x == current_event_list), axis=1)][0]
        event_labels.append(label)
        # Adds the timestamp to event_times
        event_times.append(current_earliest)
        # Resets eventlist to reflect event deletion
        eventlist = evtlist_non_empty
    return event_times, event_labels

def ProcessEvents(seg=None, tolerance=None, evtframe=None, name='Events'):
    """Takes a segment object, tolerance, and event dataframe"""
    # Makes sure that seg is a segment object
    if not isinstance(seg, neo.core.segment.Segment):
        raise TypeError('%s needs to be a neo segment object' % seg)
    # Checks if events have already been processed
    if name in [event.name for event in seg.events]:
        print("Events array has already been processed")
    else:
        # Iterates through the event object in seg and pulls out the relevent ones
        eventlist = ExtractEventsToList(seg=seg, evtframe=evtframe)
        eventtimes, eventlabels = ProcessEventList(eventlist=eventlist, 
            tolerance=tolerance, evtframe=evtframe, time_name='times', ch_name='ch')
        # Creates an Event object
        results = Event(times=np.array(eventtimes) * pq.s,
            labels=np.array(eventlabels, dtype='S'), name=name)
        # Appends event objec to segment object
        seg.events.append(results)







