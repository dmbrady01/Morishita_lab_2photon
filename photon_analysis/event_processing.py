#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
event_processing.py: Python script that contains functions for event and epoch
processing.
"""


__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "01 Mar 2018"

import collections
import six
import quantities as pq
import numbers
import neo

def EvtDict(evts=['correct', 'incorrect', 'iti_start', 'omission', 'premature', 'stimulus', 'tray']):
    """Takes in an ordered list of events and maps it to a dictionary. Keys are numbers starting at 1.
    >>> EvtDict(evts=['event1', 'event2']) == {'1': 'event1', '2': 'event2'}
    True
    """
    # Makes sure that evts is iterable (but not a string)
    if (not isinstance(evts, collections.Iterable)) or (isinstance(evts, six.string_types)):
        raise TypeError('evts needs to be iterable but not a string (list, tuple, etc.).') 
    # Calculates how many keys we need for our dictionary
    dict_len = len(evts)
    # Makes a list of numbers as our key values (but they are strings not ints!)
    keys = [str(ind + 1) for ind in range(dict_len)]
    # zips keys with our evtlist and makes it a dictionary
    return dict(zip(keys, evts))


def ProcessEvents(seg=None, tolerance=None):
    """Takes a segment object and tolerance"""
    # Makes sure that tolerance is a number
    if not isinstance(tolerance, numbers.Number):
        raise TypeError('%s needs to be a number' % tolerance)
    # Converts tolerance to a second
    tolerance = tolerance * pq.s
    # Makes sure that seg is a segment object
    if not isinstance(seg, neo.core.segment.Segment):
        raise TypeError('%s needs to be a neo segment object' % seg)
    # Checks if events have already been processed
    if 'Events' in [event.name for event in seg.events]:
        print("Events array has already been processed")
        break
    else:
        eventlist = list()
        event_times = list()
        event_labels = list()