#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_event_processing.py: Python script that contains tests for event_processing.py 
"""

__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "08 Mar 2018"

# Import unittest modules and event_processing
import unittest
from neo.core import Event
import quantities as pq
import numpy as np
import pandas as pd
from imaging_analysis.event_processing import LoadEventParams
from imaging_analysis.event_processing import TruncateEvent
from imaging_analysis.event_processing import TruncateEvents
from imaging_analysis.event_processing import ProcessEvents

class TestLoadEventParams(unittest.TestCase):
    "Code tests for LoadEventParams function"
    
    def setUp(self):
        self.dpath = 'imaging_analysis/event_params.json'
        self.evtdict = {
            'channels': ['ch1'], 
            'combinations': {
                'one': [1],
                'two': [0]
            }
        }

    def tearDown(self):
        del self.dpath
        del self.evtdict

    def test_returns_a_dataframe(self):
        "Makes sure function returns a dataframe and checks dpath works"
        self.assertIsInstance(LoadEventParams(self.dpath), pd.core.frame.DataFrame)

    def test_makes_sure_dpath_is_correct(self):
        "Makes sure dpath is correct is supplied"
        self.assertRaises(IOError, LoadEventParams, dpath='/not/path')

    def test_makes_sure_evtdict_is_correct(self):
        "Makes sure evtdict is correct is supplied"
        self.assertRaises(TypeError, LoadEventParams, evtdict='not a dict')

    def test_makes_sure_dataframe_has_right_structure(self):
        "Checks dataframe has right structure and that passing an evtdict works"
        output = LoadEventParams(evtdict=self.evtdict)
        check = pd.DataFrame(data=self.evtdict['combinations'].values(),
            index=self.evtdict['combinations'].keys(), 
            columns=self.evtdict['channels'])
        pd.testing.assert_frame_equal(output, check)



class TestTruncateEvent(unittest.TestCase):
    "Code tests for the TruncateEvent function."

    def setUp(self):
        self.evt = Event(np.arange(0, 100 ,1)*pq.s, 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.not_evt = np.random.randn(1000, 1)
        self.evt_start = 10
        self.evt_pre_start = self.evt_start - 5
        self.evt_end = 90
        self.evt_post_end = self.evt_end + 5

    def tearDown(self):
        del self.evt
        del self.not_evt
        del self.evt_start
        del self.evt_end
        del self.evt_pre_start
        del self.evt_post_end

    def test_makes_sure_events_are_passed(self):
        "Test to make sure event object is passed"
        self.assertRaises(TypeError, TruncateEvent, self.not_evt)

    def test_evt_start_works(self):
        "Test to make sure evt is truncated from start"
        trunc_sig = TruncateEvent(self.evt, start=self.evt_start)
        times = trunc_sig.times
        self.assertFalse(self.evt_pre_start in times)

    def test_evt_end_works(self):
        "Test to make sure evt is truncated from end"
        trunc_sig = TruncateEvent(self.evt, end=self.evt_end)
        times = trunc_sig.times
        self.assertFalse(self.evt_post_end in times)



class TestTruncateEvents(unittest.TestCase):
    "Code tests for TruncateEvents function."

    def setUp(self):
        self.evt = Event(np.arange(0, 100 ,1)*pq.s, 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt2 = Event(np.arange(0, 100 ,1)*pq.s, 
                        labels=np.repeat(np.array(['t2', 't3'], dtype='S'), 50))
        self.events = [self.evt, self.evt2]
        self.evt_start = 10
        self.evt_pre_start = self.evt_start - 5
        self.evt_end = 90
        self.evt_post_end = self.evt_end + 5

    def tearDown(self):
        del self.evt
        del self.evt_start
        del self.evt_end
        del self.evt_pre_start
        del self.evt_post_end
        del self.events

    def test_makes_sure_list_is_passed(self):
        "Makes sure list is passed to function"
        self.assertRaises(TypeError, TruncateEvents, 100)

    def test_evt_start_works(self):
        "Test to make sure evt is truncated from start in event_list"
        trunc_sig = TruncateEvents(self.events, start=self.evt_start)
        times = [x.times for x in trunc_sig]
        self.assertFalse(all(self.evt_pre_start in x for x in times))

    def test_evt_end_works(self):
        "Test to make sure evt is truncated from end in event_list"
        trunc_sig = TruncateEvents(self.events, end=self.evt_end)
        times = [x.times for x in trunc_sig]
        self.assertFalse(all(self.evt_post_end in x for x in times))



class TestProcessEvents(unittest.TestCase):
    "Code tests for ProcessEvents function"

    def setUp(self):
        self.wrong_type = 'hello'

    def tearDown(self):
        del self.wrong_type

    def test_tolerance_must_be_number(self):
        "Makes sure tolerance is a number"
        self.assertRaises(TypeError, ProcessEvents, tolerance=self.wrong_type)

    def test_seg_must_be_a_seg_object(self):
        "Makes sure seg is a seg object"
        self.assertRaises(TypeError, ProcessEvents, seg=self.wrong_type)

    def test_evtframe_must_be_a_dataframe_object(self):
        "Makes sure evtframe is a pandas datafrane object"
        self.assertRaises(TypeError, ProcessEvents, evtframe=self.wrong_type)



if __name__ == '__main__':
    unittest.main()