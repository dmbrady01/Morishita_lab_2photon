#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
signal_processing.py: Python script that contains functions for signal
processing.
"""


__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "07 Mar 2018"


import scipy.signal as ssp
import numpy as np
import neo
import quantities as pq

def TruncateSignal(signal, start=0, end=0):
    """Given an AnalogSignal object, will remove the first 'start' seconds and the 
    last 'end' seconds. Please note that this is different logic
    than TruncateEvent! TruncateSignals trims start/end seconds from the ends no matter
    the length. TruncateEvent start/end truncates at those specific timestamps. It
    is not relative."""
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

def ButterFilterDesign(lowcut=None, highcut=None, fs=381.469726562, order=5, 
                    btype='lowpass'):
    """Constructs the numerator and denominator for the low/high/bandpass/bandstop
    filter. low and high are the cutoff frequencies, fs is the sampling frequency, 
    order is the desired filter order, and btype designates what type of filter will
    ultimately be constructed (lowpass, highpass, bandpass, bandstop)."""
    # calculate nyquist frequency (half the sampling frequency)
    nyquist = fs * 0.5
    # try constructing normalized lowcut
    try:
        norm_low = lowcut / nyquist
    except TypeError:
        pass
    # try constructing normalized highcut
    try:
        norm_high = highcut / nyquist
        # Makes sure highcut is lower than nyquist frequency
        if norm_high > 1:
            raise ValueError('%s is larger than the nyquist frequency. Choose a smaller highcut (at most half the sampling rate).' % highcut)
    except TypeError:
        pass
    # Construct param to be bassed into butter
    if btype == 'lowpass':
        params = norm_high
    elif btype == 'highpass':
        params = norm_low
    elif (btype == 'bandpass') or (btype == 'bandstop'):
        params = [norm_low, norm_high]
    else:
        raise ValueError("%s must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'" % btype)
    # Return butter
    return ssp.butter(order, params, btype=btype, analog=False)

def FilterSignal(signal, lowcut=None, highcut=None, fs=381.469726562, order=5, 
                btype='lowpass', axis=0, window_length=3001):
    """Given some signal data, uses ButterFilterDesign or savgol_filter to 
    construct a filter and applies it to signal.
    signal: data signal
    lowcut: lower bound for filter
    highcut: upper bound for filter
    fs: sampling frequnecy
    order: filter order
    btype: 'lowpass', 'highpass', 'bandpass', 'bandstop', 'savgol'
    axis: axis of matrix to apply filter"""
    # If btype is savgol, performs a Savitzky-Golay filter
    if btype == 'savgol':
        return ssp.savgol_filter(signal, window_length, order, axis=axis)
    # Otherwise performs filfilt (backwards and forwards filtering)
    else:
        b, a = ButterFilterDesign(lowcut=lowcut, highcut=highcut, fs=fs, 
                                order=order, btype=btype)
        return ssp.filtfilt(b, a, signal, axis=axis)

def DeltaFOverF(signal, reference=None, period=None, mode='median'):
    """Calcualte DeltaF/F for a signal. There are several modes:
    'median': (signal - median(signal))/median(signal)

    'mean': (signal - mean(signal))/mean(signal)
    
    'reference': (signal - reference_signal)/reference_signal
    Note that reference must be a same length array as signal
    
    'period_mean': ref = mean(signal[period[0]:period[1]] 
                   (signal - ref)/ref
    
    'period_median': same as 'period_mean' but uses median instead
    Note that period must be a list or tuple of beginning and end timstamps"""
    if mode == 'median':
        return (signal - np.median(signal))/np.median(signal)
    elif mode == 'mean':
        return (signal - np.mean(signal))/np.mean(signal)
    elif mode == 'reference':
        return (signal - reference)/reference
    elif mode == 'period_mean':
        reference = signal[period[0]:period[1]]
        return (signal - np.mean(reference))/np.mean(reference)
    elif mode == 'period_median':
        reference = signal[period[0]:period[1]]
        return (signal - np.median(reference))/np.median(reference)
    else:
        raise ValueError('%s is not an accepted mode for calculating deltaf' % mode)


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