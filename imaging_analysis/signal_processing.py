#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
signal_processing.py: Python script that contains functions for signal
processing.
"""


__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "06 Mar 2018"


import scipy.signal as ssp
import numpy as np
import neo
import quantities as pq

def TruncateSignal(signal, start=0, end=0):
    """Given an AnalogSignal object, will remove the first 'start' seconds and the 
    last 'end' seconds."""
    # Makes sure signal is an AnalogSignal object
    if not isinstance(signal, neo.core.AnalogSignal):
        raise TypeError('%s must be an AnalogSignal object.' % signal)
    # converts start and end to times
    start = start * pq.s 
    end = end * pq.s
    # Truncate signal
    truncated_signal = signal.time_slice(signal.t_start + start, signal.t_stop - end)

    return truncated_signal


def TruncateSignals(signal_list, start=0, end=0):
    """Given a list of AnalogSignal objects, will iterate through each one and remove
    values before 'start' and after 'end'. Start and end must be in seconds."""
    # Makes sure a list is passed
    if not isinstance(signal_list, list):
        raise TypeError('%s must be a list' % signal_list)
    # Iterate through each item with TrunacteSignal
    truncated_list = [TruncateSignal(sig, start=start, end=end) for sig in signal_list]
    
    return truncated_list


# def NormalizeSignal(signal=None, reference=None, framelen=3001, order=1, return_filt=False):
#     """Given a signal and a reference, it returns the signal - savgol_filter(reference).
#     If only a signal is given, it returns the signal - savgol_filter(signal). 
#     If return_filt = True, then we just return the filtered signal or reference."""
#     # We determine which axis the signal is recorded (column or row vector)
#     if signal.shape[0] > signal.shape[1]:
#         axis = 0
#     else:
#         axis = 1
#     # We filter the signal with a Savtizky-Golay filter
#     if reference:
#         filtered_signal = ssp.savgol_filter(reference, framelen, order, axis=axis)
#     else:
#         filtered_signal = ssp.savgol_filter(signal, framelen, order, axis=axis)
#     # Returns the signal - filtered signal (or just the filtered signal)
#     if return_filt:
#         return filtered_signal
#     else:
#         return signal - filtered_signal


# def SubtractNoise(signal=None, reference=None, framelen=3001, order=1):
#     """Given a signal and reference, it subtracts the filtered reference from
#     the signal. Both signals are median subtracted/scaled first."""
#     # First we center and scale the signal/reference by the median
#     median_signal = (signal - np.median(signal))/np.median(signal)
#     median_reference = (reference - np.median(reference))/np.median(reference)
#     # Then we filter the reference with a Savtizky-Golay filter
#     normalized_sig = NormalizeSignal(median_signal, median_reference, framelen, order)
#     # We return the median_subtracted_signal - the filtered_reference
#     return median_signal - filt_reference




# def NormalizeSignalData(seg=None, signal=['LMag 1'], reference=['LMag 2']):
#     pass