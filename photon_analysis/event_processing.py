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