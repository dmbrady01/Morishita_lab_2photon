#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_signal_processing.py: Python script that contains tests for signal_processing.py 
"""

__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "06 Mar 2018"

# Import unittest modules and event_processing
import unittest
from neo.core import AnalogSignal
import numpy as np
import quantities as pq
from imaging_analysis.signal_processing import TruncateSignal
from imaging_analysis.signal_processing import TruncateSignals

class TestTruncateSignal(unittest.TestCase):
    "Tests for the TruncateSignal function."

    def setUp(self):
        self.signal = AnalogSignal(np.random.randn(1000,1), units='V', sampling_rate=1*pq.Hz)
        self.not_analog = np.random.randn(1000, 1)
        self.signal_start = 10
        self.signal_end = 10

    def tearDown(self):
        del self.signal
        del self.not_analog
        del self.signal_start
        del self.signal_end

    def test_makes_sure_analog_signal_is_passed(self):
        "Test to make sure analog signal object is passed"
        self.assertRaises(TypeError, TruncateSignal, self.not_analog)

    def test_signal_start_works(self):
        "Test to make sure start is truncated"
        trunc_sig = TruncateSignal(self.signal, start=self.signal_start)
        self.assertEqual(trunc_sig.t_start, self.signal_start * pq.s)

    def test_signal_end_works(self):
        "Test to make sure start is truncated"
        trunc_sig = TruncateSignal(self.signal, end=self.signal_end)
        true_stop = self.signal.t_stop - self.signal_end * pq.s
        self.assertEqual(trunc_sig.t_stop, true_stop)



class TestTruncateSignals(unittest.TestCase):
    "Code tests for TruncateSignals function."

    def setUp(self):
        self.signal = AnalogSignal(np.random.randn(1000,1), units='V', sampling_rate=1*pq.Hz)
        self.signal2 = AnalogSignal(np.random.randn(900,1), units='V', sampling_rate=1*pq.Hz)
        self.signals = [self.signal, self.signal2]
        self.signal_start = 10
        self.signal_end = 10

    def tearDown(self):
        del self.signal
        del self.signal2
        del self.signals
        del self.signal_start
        del self.signal_end

    def test_makes_sure_list_is_passed(self):
        "Makes sure list is passed to function"
        self.assertRaises(TypeError, TruncateSignals, 100)

    def test_signal_start_works(self):
        "Test to make sure signal is truncated from start in signal_list"
        trunc_sig = TruncateSignals(self.signals, start=self.signal_start)
        t_starts = [x.t_start for x in trunc_sig]
        self.assertTrue(all(self.signal_start * pq.s == x for x in t_starts))

    def test_signal_end_works(self):
        "Test to make sure signal is truncated from end in signal_list"
        trunc_sig = TruncateSignals(self.signals, end=self.signal_end)
        t_stops = [x.t_stop for x in trunc_sig]
        true_stops = [x.t_stop - self.signal_end * pq.s for x in self.signals]
        paired = zip(t_stops, true_stops)
        self.assertTrue(all(x == y for x, y in paired))



if __name__ == '__main__':
    unittest.main()