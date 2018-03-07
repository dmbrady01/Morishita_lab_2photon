#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_signal_processing.py: Python script that contains tests for signal_processing.py 
"""

__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "07 Mar 2018"

# Import unittest modules and event_processing
import unittest
from neo.core import AnalogSignal
import numpy as np
import quantities as pq
import scipy.signal as ssp
from imaging_analysis.signal_processing import TruncateSignal
from imaging_analysis.signal_processing import TruncateSignals
from imaging_analysis.signal_processing import ButterFilterDesign
from imaging_analysis.signal_processing import FilterSignal

class TestTruncateSignal(unittest.TestCase):
    "Tests for the TruncateSignal function."

    def setUp(self):
        self.signal = AnalogSignal(np.random.randn(1000,1), units='V',
                                    sampling_rate=1*pq.Hz)
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



class TestButterFilterDesign(unittest.TestCase):
    "Code tests for ButterFilterDesign."

    def setUp(self):
        self.order = 10
        self.fs = 30
        self.high = 10
        self.low = .5
        self.wrong_btype = 'hamburger'

    def tearDown(self):
        del self.order
        del self.fs
        del self.low
        del self.high
        del self.wrong_btype

    def test_make_sure_correct_btype_is_passed(self):
        "Makes sure btype must be correct"
        self.assertRaises(ValueError, ButterFilterDesign, btype=self.wrong_btype)

    def test_make_sure_highcut_is_below_nyquist(self):
        "Makes sure highcut is below nyquist frequency"
        self.assertRaises(ValueError, ButterFilterDesign, highcut=self.fs, fs=self.fs)

    def test_lowpass_filter_design_works(self):
        "Makes sure lowpass filter design works"
        output = ButterFilterDesign(highcut=self.high, fs=self.fs, order=self.order, 
                                btype='lowpass')
        check = ssp.butter(self.order, self.high/(self.fs*.5), btype='lowpass')
        self.assertTrue(np.array_equal(output, check))

    def test_highpass_filter_design_works(self):
        "Makes sure highpass filter design works"
        output = ButterFilterDesign(lowcut=self.low, fs=self.fs, order=self.order, 
                                btype='highpass')
        check = ssp.butter(self.order, self.low/(self.fs*.5), btype='highpass')
        self.assertTrue(np.array_equal(output, check))

    def test_bandpass_filter_design_works(self):
        "Makes sure bandpass filter design works"
        output = ButterFilterDesign(lowcut=self.low, highcut=self.high, fs=self.fs, 
                                order=self.order, btype='bandpass')
        check = ssp.butter(self.order, 
                            [self.low/(self.fs*.5), self.high/(self.fs*.5)], 
                            btype='bandpass')
        self.assertTrue(np.array_equal(output, check))

    def test_bandstop_filter_design_works(self):
        "Makes sure bandstop filter design works"
        output = ButterFilterDesign(lowcut=self.low, highcut=self.high, fs=self.fs, 
                                order=self.order, btype='bandstop')
        check = ssp.butter(self.order, 
                            [self.low/(self.fs*.5), self.high/(self.fs*.5)], 
                            btype='bandstop')
        self.assertTrue(np.array_equal(output, check))



class TestFilterSignal(unittest.TestCase):
    "Code tests for FilterSignal function."

    def setUp(self):
        self.T = 5
        self.fs = 30
        self.nsamples = int(self.T * self.fs)
        self.t = np.linspace(0, self.T, self.nsamples, endpoint=False)
        self.signal = np.sin(1.2*2*np.pi*self.t) + 1.5*np.cos(9*2*np.pi*self.t) + \
                            0.5*np.sin(12.0*2*np.pi*self.t)
        self.highcut = 3.667
        self.lowcut = 0.5
        self.savgol_order = 1
        self.order = 5
        self.window_length = 11

    def tearDown(self):
        del self.T 
        del self.fs
        del self.nsamples
        del self.t
        del self.signal 
        del self.highcut
        del self.lowcut
        del self.savgol_order
        del self.order
        del self.window_length

    def test_lowpass_works(self):
        "Tests lowpass functionality"
        signal = FilterSignal(self.signal, lowcut=self.lowcut, highcut=self.highcut, 
                            fs=self.fs, order=self.order, btype='lowpass')
        b, a = ButterFilterDesign(lowcut=self.lowcut, highcut=self.highcut, 
                                        fs=self.fs, order=self.order, btype='lowpass')
        test_signal = ssp.filtfilt(b, a, self.signal)
        equal = np.array_equal(signal, test_signal)
        self.assertTrue(equal)

    def test_highpass_works(self):
        "Tests highpass functionality"
        signal = FilterSignal(self.signal, lowcut=self.lowcut, highcut=self.highcut, 
                            fs=self.fs, order=self.order, btype='highpass')
        b, a = ButterFilterDesign(lowcut=self.lowcut, highcut=self.highcut, 
                                        fs=self.fs, order=self.order, btype='highpass')
        test_signal = ssp.filtfilt(b, a, self.signal)
        equal = np.array_equal(signal, test_signal)
        self.assertTrue(equal)    

    def test_bandpass_works(self):
        "Tests bandpass functionality"
        signal = FilterSignal(self.signal, lowcut=self.lowcut, highcut=self.highcut, 
                            fs=self.fs, order=self.order, btype='bandpass')
        b, a = ButterFilterDesign(lowcut=self.lowcut, highcut=self.highcut, 
                                        fs=self.fs, order=self.order, btype='bandpass')
        test_signal = ssp.filtfilt(b, a, self.signal)
        equal = np.array_equal(signal, test_signal)
        self.assertTrue(equal) 

    def test_bandstop_works(self):
        "Tests bandpass functionality"
        signal = FilterSignal(self.signal, lowcut=self.lowcut, highcut=self.highcut, 
                            fs=self.fs, order=self.order, btype='bandstop')
        b, a = ButterFilterDesign(lowcut=self.lowcut, highcut=self.highcut, 
                                        fs=self.fs, order=self.order, btype='bandstop')
        test_signal = ssp.filtfilt(b, a, self.signal)
        equal = np.array_equal(signal, test_signal)
        self.assertTrue(equal) 

    def test_savgol_works(self):
        "Tests savgol functionality"
        signal = FilterSignal(self.signal, order=self.savgol_order, 
                            window_length=self.window_length, btype='savgol')
        test_signal = ssp.savgol_filter(self.signal, self.window_length, 
                                        self.savgol_order, axis=0)
        equal = np.array_equal(signal, test_signal)
        self.assertTrue(equal)


if __name__ == '__main__':
    unittest.main()