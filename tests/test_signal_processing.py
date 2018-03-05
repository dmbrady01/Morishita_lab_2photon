#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_signal_processing.py: Python script that contains tests for signal_processing.py 
"""

__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "05 Mar 2018"

# Import unittest modules and event_processing
import unittest
from neo.core import AnalogSignal
import numpy as np
import quantities as pq
from imaging_analysis.signal_processing import TruncateSignal

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







if __name__ == '__main__':
    unittest.main()