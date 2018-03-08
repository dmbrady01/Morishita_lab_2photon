#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_signal_processing.py: Python script that contains tests for signal_processing.py 
"""

__author__ = "DM Brady"
__datewritten__ = "06 Mar 2018"
__lastmodified__ = "06 Mar 2018"

# Import unittest modules and event_processing
import unittest
from neo.core import AnalogSignal, Event, Segment
import numpy as np
import quantities as pq
import copy as cp
from imaging_analysis.segment_processing import TruncateSegment
from imaging_analysis.segment_processing import TruncateSegments


class TestTruncateSegment(unittest.TestCase):
    "Tests for the TruncateSegment function."

    def setUp(self):
        self.signal = AnalogSignal(np.random.randn(1000,1), units='V', 
            sampling_rate=1*pq.Hz)
        self.signal_start = 10
        self.signal_end = 10
        self.evt = Event(np.arange(0, 100 ,1)*pq.s, 
            labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt_start = 15
        self.evt_pre_start = self.evt_start - 5
        self.evt_end = 85
        self.evt_post_end = self.evt_end + 5
        self.not_segment = [100]
        self.segment = Segment()
        self.segment.analogsignals.append(self.signal)
        self.segment.events.append(self.evt)

    def tearDown(self):
        del self.signal
        del self.signal_start
        del self.signal_end
        del self.evt
        del self.evt_start
        del self.evt_pre_start
        del self.evt_end
        del self.evt_post_end
        del self.not_segment
        del self.segment

    def test_makes_sure_segment_is_passed(self):
        "Test to make sure segment object is passed"
        self.assertRaises(TypeError, TruncateSegment, self.not_segment)

    def test_signals_are_truncated(self):
        "Test to make sure analog starts/ends are truncated"
        trunc_seg = TruncateSegment(self.segment, start=self.signal_start, 
            end=self.signal_end)
        analog = trunc_seg.analogsignals[0]
        true_start = self.signal_start * pq.s
        true_stop = self.signal.t_stop - self.signal_end * pq.s
        test_start = true_start == analog.t_start
        test_stop = true_stop == analog.t_stop
        self.assertTrue(all([test_start, test_stop]))

    def test_events_are_truncated_the_same_if_clip_same(self):
        "Test to make sure analog starts/ends are truncated"
        trunc_seg = TruncateSegment(self.segment, start=self.signal_start, 
            end=self.signal_end, clip_same=True)
        event_times = trunc_seg.events[0].times
        test_start = self.signal_start - 1 not in event_times
        true_stop = self.signal.t_stop - self.signal_end * pq.s
        test_stop = true_stop + 1 * pq.s not in event_times
        self.assertTrue(all([test_start, test_stop]))

    def test_events_are_truncated_differently_if_clip_not_same(self):
        "Test to make sure analog starts/ends are truncated"
        trunc_seg = TruncateSegment(self.segment, start=self.signal_start, 
            end=self.signal_end, clip_same=False, evt_start=self.evt_start, 
            evt_end=self.evt_end)
        event_times = trunc_seg.events[0].times
        test_start = self.evt_pre_start not in event_times
        test_stop = self.evt_post_end not in event_times
        self.assertTrue(all([test_start, test_stop]))



class TestTruncateSegments(unittest.TestCase):
    "Tests for the TruncateSegments function."

    def setUp(self):
        self.signal = AnalogSignal(np.random.randn(1000,1), units='V', 
            sampling_rate=1*pq.Hz)
        self.signal2 = AnalogSignal(np.random.randn(1000,1), units='V', 
            sampling_rate=1*pq.Hz)
        self.signal_start = 10
        self.signal_end = 10
        self.evt = Event(np.arange(0, 100 ,1)*pq.s, 
            labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt2 = Event(np.arange(0, 100 ,1)*pq.s, 
            labels=np.repeat(np.array(['t2', 't3'], dtype='S'), 50))
        self.evt_start = 15
        self.evt_pre_start = self.evt_start - 5
        self.evt_end = 85
        self.evt_post_end = self.evt_end + 5
        self.not_segment = [100]
        self.segment = Segment()
        self.segment.analogsignals.append(self.signal)
        self.segment.events.append(self.evt)
        self.segment2 = Segment()
        self.segment2.analogsignals.append(self.signal2)
        self.segment2.events.append(self.evt2)       
        self.segments = [self.segment, self.segment2]

    def tearDown(self):
        del self.signal
        del self.signal_start
        del self.signal_end
        del self.evt
        del self.evt_start
        del self.evt_pre_start
        del self.evt_end
        del self.evt_post_end
        del self.not_segment
        del self.segment
        del self.segments

    def test_makes_sure_segment_is_passed(self):
        "Test to make sure list object is passed"
        self.assertRaises(TypeError, TruncateSegments, self.segment)

    def test_signals_are_truncated(self):
        "Test to make sure analog starts/ends are truncated for every segment"
        trunc_segs = TruncateSegments(self.segments, start=self.signal_start, 
            end=self.signal_end)
        choice_to_test = np.random.choice(range(len(trunc_segs)))
        analog = trunc_segs[choice_to_test].analogsignals[0]
        true_start = self.signal_start * pq.s
        true_stop = self.signal.t_stop - self.signal_end * pq.s
        test_start = true_start == analog.t_start
        test_stop = true_stop == analog.t_stop
        self.assertTrue(all([test_start, test_stop]))

    def test_events_are_truncated_the_same_if_clip_same(self):
        "Test to make sure analog starts/ends are truncated for every segment"
        trunc_segs = TruncateSegments(self.segments, start=self.signal_start, 
            end=self.signal_end, clip_same=True)
        choice_to_test = np.random.choice(range(len(trunc_segs)))
        event_times = trunc_segs[choice_to_test].events[0].times
        test_start = self.signal_start - 1 not in event_times
        true_stop = self.signal.t_stop - self.signal_end * pq.s
        test_stop = true_stop + 1 * pq.s not in event_times
        self.assertTrue(all([test_start, test_stop]))

    def test_events_are_truncated_differently_if_clip_not_same(self):
        "Test to make sure analog starts/ends are truncated"
        trunc_segs = TruncateSegments(self.segments, start=self.signal_start, 
            end=self.signal_end, clip_same=False, evt_start=self.evt_start, 
            evt_end=self.evt_end)
        choice_to_test = np.random.choice(range(len(trunc_segs)))
        event_times = trunc_segs[choice_to_test].events[0].times
        test_start = self.evt_pre_start not in event_times
        test_stop = self.evt_post_end not in event_times
        self.assertTrue(all([test_start, test_stop]))



if __name__ == '__main__':
    unittest.main()